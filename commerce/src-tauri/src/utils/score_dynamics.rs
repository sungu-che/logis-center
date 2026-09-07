// =====================================================================
// 🌟 [SCORE DYNAMICS LAYER] 점수를 스칼라가 아니라 신호로 취급하는 계층
// ---------------------------------------------------------------------
//  ── 이 모듈이 존재하는 이유 ──
//   현재 파이프라인의 모든 판정(비전 히트맵, 문서 type 분류, PLINKO 배정,
//   청크 인덱싱, analytic 쿼리 파싱)은 코사인 점수 하나를 스칼라로 읽고
//   고정 상수와 비교하는 '횡단적(cross-sectional)' 구조입니다.
//
//   그래서 다음 세 가지가 구조적으로 불가능합니다.
//     L1 절대 점수의 의미가 문서마다 다른데 기준이 고정
//        → TITLE FLOOR 0.5457 바닥 미달로 'Shipping Advice' 표제가 탈락,
//          제목 축 정보가 0 으로 수렴
//     L2 1위-2위 단일 마진만 보고 확신도를 판정
//        → MODE PROBE 마진 +0.0308 (잡음대 0.5334) 코인플립,
//          PLINKO 'related_po_number'→'marks_numbers' 마진 +0.0001
//     L3 편향 보정 상수가 문서·사이트 간에 적응하지 않음
//        → CATEGORY-NEUTRAL cargo(-1.973) / financials(-2.297) 고정,
//          HEATMAP 활성 패치 249/252 과확산
//
//  ── 이 모듈이 하는 일 / 하지 않는 일 ──
//   합니다   : 판정 근거를 구조화 관측으로 수집(SSR), 요약 통계로 압축해
//              세션 간 영속화(SDS), 적응 파라미터 조회 인터페이스 제공(ASE)
//   안 합니다: 예측. 신경망. 판정 변경.
//              Phase 0 에서 ASE 조회 함수는 정의만 되고 호출되지 않습니다.
//
//  ── 왜 신경망 시계열 모델이 아닌가 ──
//   ① 문서 추출에는 물리적 시간이 없어 순위/인덱스를 시간으로 치환해야 하고,
//      그 계열의 길이가 3~15보라 학습형 모델은 과적합합니다.
//   ② 온디바이스 VRAM 에서 Qwen3.5(2B) / SigLIP2(2.2GB) / granite(97M) 과
//      경쟁해야 하므로 추가 가중치는 ROI 가 음수입니다.
//   따라서 도구는 Welford 온라인 통계 · 분포 형상 요약 · 역분산 가중이며,
//   전부 순수 산술이라 모델 로드가 없습니다.
//
//  ── 기존 철학과의 정합 ──
//   이 코드베이스는 이미 '기준선은 데이터에서 유도' 를 채택하고 있습니다.
//     scheduler.rs   dedup_floor = μ + 3σ
//     trading.rs     TITLE FLOOR = 자기선언 분포 평균
//     ai_utils.rs    pooled σ, gumbel_expected_z(N)
//     vision_encoder score_patches_bank_neutral 의 패치별 μ_k
//   정적 상수가 남아 있는 곳은 '문서 간 축' 뿐이며, 이 모듈은 그 축 하나를
//   메우는 것입니다. bias.json 이 정적 사전이라면 이것은 동적 사전입니다.
// =====================================================================
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

// =====================================================================
// 🌟 [세대 각인] 관측 통계는 '같은 앵커 뱅크 · 같은 임베딩 레시피' 아래에서만
//    의미가 있습니다. bias.json 이 개정되거나 임베딩 모델이 바뀌면
//    과거 점수 분포는 현재 판정과 무관한 숫자가 됩니다.
//    store.rs 의 SCHEMA_VERSION / embed_recipe_v3 와 동일한 방어입니다.
//    이 문자열이 달라지면 저장된 통계를 전량 폐기하고 새로 시작합니다.
// =====================================================================
pub const SDS_RECIPE: &str = "sds-v1:welford+ring/granite-384/bias-json";

// =====================================================================
// 🌟 [최소 관측 수] 기획 6-2 의 냉간 시작 임계값입니다.
//
//  ── 왜 이것은 '매직 상수' 가 아닌가 ──
//   이 값은 판정에 쓰이는 임계치가 아니라 '통계를 신뢰할 수 있는 표본 수'
//   입니다. 미달이면 ASE 는 None 을 돌려주고 호출부는 기존 상수를 그대로
//   씁니다. 즉 이 값이 틀려도 잘못된 판정이 생기지 않고,
//   적응이 늦게 시작될 뿐입니다.
//
//  ── 링 버퍼 크기로도 재사용 ──
//   최근 구간 평균과 전체 평균의 비교로 드리프트를 보려면 창이 필요한데,
//   그 창 크기를 여기서 그대로 가져다 씁니다. 새 상수를 만들지 않기 위함입니다.
// =====================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    /// 이미지 문서 전처리 (SigLIP2 비전 경로)
    Vision,
    /// 텍스트 문서 전처리 (트레이딩 경로)
    Trading,
    /// 커머스 목록/상세 및 청크 인덱싱
    Commerce,
    /// 웹사이트 사용자 의도 분석
    Analytic,
}

impl Track {
    pub fn as_str(&self) -> &'static str {
        match self {
            Track::Vision => "vision",
            Track::Trading => "trading",
            Track::Commerce => "commerce",
            Track::Analytic => "analytic",
        }
    }
    /// (2차 스코프 발동, 1차 스코프 발동, 전역 발동)
    pub fn min_obs(&self) -> (u64, u64, u64) {
        match self {
            Track::Vision => (20, 8, 30),
            Track::Trading => (30, 12, 40),
            Track::Commerce => (50, 20, 60),
            Track::Analytic => (40, 15, 0),
        }
    }
    /// 링 버퍼 크기 = 1차 스코프 발동 수. 새 상수를 만들지 않기 위한 재사용입니다.
    pub fn ring_len(&self) -> usize {
        self.min_obs().1 as usize
    }
}

// =====================================================================
// 🌟 [SCOPE] 통계 격리 단위
// ---------------------------------------------------------------------
//  너무 넓으면 서로 다른 레이아웃이 섞여 평균이 무의미해지고,
//  너무 좁으면 표본이 모이지 않습니다. 그래서 2단으로 두고 폴백합니다.
//
//    Vision   1차 doc_type              2차 doc_type × 발행처 해시
//    Trading  1차 mode × doc_type       2차 mode × doc_type × cc
//    Commerce 1차 cc × page_type        2차 cc × page_type × detail
//    Analytic 1차 cc                    2차 cc × ref
// =====================================================================
#[derive(Debug, Clone, Default)]
pub struct Scope {
    pub team: String,
    pub track_name: String,
    pub primary: String,
    pub secondary: String,
}

impl Scope {
    pub fn key_secondary(&self) -> String {
        format!("{}|{}|{}", self.track_name, self.primary, self.secondary)
    }
    pub fn key_primary(&self) -> String {
        format!("{}|{}|", self.track_name, self.primary)
    }
    pub fn key_global(&self) -> String {
        format!("{}||", self.track_name)
    }
    pub fn is_empty(&self) -> bool {
        self.track_name.is_empty()
    }
}

// =====================================================================
// 🌟 [ACTIVE SCOPE] 태스크 진입 시 세우는 전역 스코프
// ---------------------------------------------------------------------
//  ── 왜 전역인가 ──
//   vision_encoder::classify_doc_type / build_column_heatmaps 는 순수 함수라
//   team_id 를 갖지 않습니다. 인자로 스코프를 밀어 넣으면 시그니처가 전부
//   바뀌고 호출부가 광범위하게 흔들립니다.
//   태스크는 모델 락(model_mutex)으로 직렬화되므로 동시에 두 스코프가
//   활성화되지 않으며, CROSSOVER_PHASE / TRANSLIT_MEM_CACHE 가 같은 선례입니다.
//
//  ── 스코프가 비어 있으면 ──
//   모든 record_* 함수가 즉시 반환합니다. 계측 누락이지 오류가 아닙니다.
// =====================================================================
static ACTIVE_SCOPE: Lazy<RwLock<Scope>> = Lazy::new(|| RwLock::new(Scope::default()));

pub fn enter_scope(team: &str, track: Track, primary: &str, secondary: &str) {
    if let Ok(mut w) = ACTIVE_SCOPE.write() {
        *w = Scope {
            team: team.to_string(),
            track_name: track.as_str().to_string(),
            primary: primary.trim().to_lowercase(),
            secondary: secondary.trim().to_lowercase(),
        };
    }
}

/// 스코프의 1차 키만 나중에 확정되는 경우(문서 type 을 판정한 직후 등)를 위한 갱신입니다.
pub fn refine_primary(primary: &str) {
    if let Ok(mut w) = ACTIVE_SCOPE.write() {
        if !w.track_name.is_empty() {
            w.primary = primary.trim().to_lowercase();
        }
    }
}

pub fn leave_scope() {
    if let Ok(mut w) = ACTIVE_SCOPE.write() {
        *w = Scope::default();
    }
}

fn current_scope() -> Option<Scope> {
    let s = ACTIVE_SCOPE.read().ok()?.clone();
    if s.is_empty() { None } else { Some(s) }
}

// 🌟 [UNSCOPED FALLBACK] 스코프가 없어도 관측을 버리지 않습니다.
// ---------------------------------------------------------------------
//  ── 왜 필요한가 (실측 사고) ──
//   초기 설계는 스코프가 없으면 with_scope_mut 이 조용히 return 했습니다.
//   그래서 enter_scope 배치를 한 군데라도 틀리면
//     · 모든 record_* 가 아무 소리 없이 버려지고
//     · DIRTY 가 false 로 남아 flush 가 파일을 쓰지 않으며
//     · 로그에는 아무 흔적도 남지 않습니다.
//   실제로 process_task 의 shipping/image 조기 return 을 놓쳐
//   두 경로 모두 계측이 통째로 죽었는데 이를 알아챌 방법이 없었습니다.
//   '실패가 보이지 않는 설계' 는 그 자체가 결함입니다.
//
//  ── 어떻게 고치는가 ──
//   스코프가 없으면 'unscoped' 라는 명시적 스코프에 기록합니다.
//   통계로서의 가치는 낮지만(서식별 격리가 안 됨),
//     ① 파일이 생성되어 배선 성공을 즉시 확인할 수 있고
//     ② 아래 경고 로그가 '어디서 스코프가 비었는지' 를 알려줍니다.
//   즉 조용한 유실이 시끄러운 진단으로 바뀝니다.
//
//  ── 경고를 1회만 내는 이유 ──
//   record_* 는 문서 1건에 수백 회 호출됩니다.
//   매번 찍으면 로그가 묻히므로 프로세스당 1회만 알립니다.
static UNSCOPED_WARNED: Lazy<RwLock<bool>> = Lazy::new(|| RwLock::new(false));

fn effective_scope_key() -> (String, usize) {
    match (current_scope(), current_track()) {
        (Some(s), Some(t)) => (s.key_secondary(), t.ring_len()),
        _ => {
            let already = UNSCOPED_WARNED.read().map(|w| *w).unwrap_or(true);
            if !already {
                if let Ok(mut w) = UNSCOPED_WARNED.write() { *w = true; }
                println!(
                    "[SDS] ⚠️ 활성 스코프 없이 관측이 들어왔습니다. 'unscoped' 로 기록합니다. \
                     enter_scope 호출부가 누락된 경로가 있습니다 — 통계는 남지만 서식별 격리가 되지 않습니다. \
                     (이 경고는 프로세스당 1회만 출력됩니다)"
                );
            }
            ("unscoped||".to_string(), 12usize)
        }
    }
}

fn current_track() -> Option<Track> {
    let s = current_scope()?;
    Some(match s.track_name.as_str() {
        "vision" => Track::Vision,
        "trading" => Track::Trading,
        "commerce" => Track::Commerce,
        "analytic" => Track::Analytic,
        _ => return None,
    })
}

// =====================================================================
// 🌟 [WELFORD] 온라인 평균/분산
// ---------------------------------------------------------------------
//  ── 왜 EWMA 가 아닌가 ──
//   EWMA 의 α 는 그 자체가 매직 상수입니다. 기획 1-5 의 '새 매직 상수 금지'
//   원칙에 걸립니다. Welford 는 상수가 없고 수치적으로 안정하며,
//   n·mean·m2 세 값만 저장하면 되어 파일이 커지지 않습니다.
//   드리프트는 아래 ring 과 전체 평균의 비교로 봅니다.
// =====================================================================
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Welford {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    /// 최근 K개 관측. 드리프트 판정용이며 K = Track::ring_len()
    #[serde(default)]
    pub ring: Vec<f64>,
}

impl Welford {
    pub fn push(&mut self, x: f64, ring_len: usize) {
        if !x.is_finite() { return; }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / (self.n as f64);
        self.m2 += d * (x - self.mean);
        self.ring.push(x);
        if ring_len > 0 && self.ring.len() > ring_len {
            let excess = self.ring.len() - ring_len;
            self.ring.drain(0..excess);
        }
    }
    pub fn variance(&self) -> f64 {
        if self.n < 2 { 0.0 } else { self.m2 / ((self.n - 1) as f64) }
    }
    pub fn sd(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }
    /// 최근 창 평균. 링이 비면 전체 평균으로 폴백합니다.
    pub fn recent_mean(&self) -> f64 {
        if self.ring.is_empty() { return self.mean; }
        self.ring.iter().sum::<f64>() / (self.ring.len() as f64)
    }
    /// 드리프트 z 값. 전체 평균 대비 최근 창 평균이 몇 σ 떨어져 있는가.
    pub fn drift_z(&self) -> f64 {
        let sd = self.sd();
        if sd <= 0.0 || self.ring.is_empty() { return 0.0; }
        (self.recent_mean() - self.mean) / sd
    }
}

// =====================================================================
// 🌟 [DECAY PROFILE] 순위 감쇠 곡선의 형상 요약
// ---------------------------------------------------------------------
//  ── 무엇을 해결하려는 통계인가 ──
//   현재 판정은 1위-2위 마진만 봅니다. 그래서 다음 두 상황을 구분할 수 없습니다.
//     승자독식형 [6.28, 2.42, 2.03, 2.02] — 낮은 마진이어도 1위가 이상치
//     평탄형     [0.60, 0.59, 0.58, 0.57] — 높은 마진이어도 변별력 없음
//   실측 로그 `[VISION CODE MARGIN] 마진 +0.0434 | 양수 점수 코드: 55/55` 는
//   전형적인 평탄형이며, 마진만으로는 이 사실이 드러나지 않습니다.
//
//  ── 저장하는 형상 지표 ──
//   top_gap_ratio : (1위-2위) / (1위-꼬리중앙값). 1 에 가까울수록 승자독식
//   tail_flatness : 2위 이하의 표준편차 / 전체 범위. 0 에 가까울수록 평탄
//   entropy_norm  : softmax 정규화 엔트로피 / ln(N). 1 에 가까울수록 무변별
//   positive_ratio: 양수 점수 비율. 1.0 이면 전 후보가 살아남은 상태
// =====================================================================
#[derive(Debug, Clone, Copy)]
pub struct DecayShape {
    pub n: usize,
    pub top1: f64,
    pub top2: f64,
    pub margin: f64,
    pub top_gap_ratio: f64,
    pub tail_flatness: f64,
    pub entropy_norm: f64,
    pub positive_ratio: f64,
}

/// 정렬 여부와 무관하게 점수 배열에서 감쇠 형상을 산출합니다.
pub fn decay_shape(scores: &[f32]) -> Option<DecayShape> {
    let mut v: Vec<f64> = scores
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| *s as f64)
        .collect();
    if v.len() < 2 { return None; }
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let top1 = v[0];
    let top2 = v[1];
    let margin = top1 - top2;

    // 꼬리 중앙값: 2위 이하의 중앙값. 1위가 얼마나 떨어져 있는지의 기준선입니다.
    let tail = &v[1..];
    let tail_median = tail[tail.len() / 2];
    let span = (top1 - tail_median).abs();
    let top_gap_ratio = if span > 1e-9 { (margin / span).clamp(0.0, 1.0) } else { 0.0 };

    // 꼬리 평탄도
    let tail_mean = tail.iter().sum::<f64>() / (tail.len() as f64);
    let tail_var = tail.iter().map(|x| (x - tail_mean).powi(2)).sum::<f64>() / (tail.len() as f64);
    let full_range = (v[0] - v[n - 1]).abs();
    let tail_flatness = if full_range > 1e-9 {
        (tail_var.sqrt() / full_range).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // 정규화 엔트로피. 척도 불변을 위해 범위로 스케일링한 뒤 softmax 를 씁니다.
    let entropy_norm = {
        let scale = if full_range > 1e-9 { full_range } else { 1.0 };
        let mx = v[0];
        let exps: Vec<f64> = v.iter().map(|x| ((x - mx) / scale).exp()).collect();
        let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
        let mut h = 0.0f64;
        for e in exps.iter() {
            let p = e / sum;
            if p > 1e-12 { h -= p * p.ln(); }
        }
        let hmax = (n as f64).ln();
        if hmax > 1e-9 { (h / hmax).clamp(0.0, 1.0) } else { 0.0 }
    };

    let positive_ratio = v.iter().filter(|x| **x > 0.0).count() as f64 / (n as f64);

    Some(DecayShape {
        n,
        top1,
        top2,
        margin,
        top_gap_ratio,
        tail_flatness,
        entropy_norm,
        positive_ratio,
    })
}

// =====================================================================
// 🌟 [저장 레코드] 기획 6-1 의 통계 레코드 정의를 그대로 옮긴 구조입니다.
// =====================================================================
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecayStat {
    pub margin: Welford,
    pub top_gap_ratio: Welford,
    pub tail_flatness: Welford,
    pub entropy_norm: Welford,
    pub positive_ratio: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldRejectStat {
    /// 후보로 검토된 횟수
    pub seen: u64,
    pub reject_format: u64,
    pub reject_prejudice: u64,
    pub reject_enum: u64,
    pub reject_self_id: u64,
    /// 니어미스(CONFIRM FLAG) 발생 횟수
    pub near_miss: u64,
    /// 최종 확정 횟수
    pub assigned: u64,
    /// 확정 시 마진 분포
    pub assign_margin: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusionStat {
    pub ties: u64,
    pub a_wins: u64,
    pub b_wins: u64,
    pub margin: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryStat {
    /// 스키마상 필드 수 (구조가 가정한 드로잉 수 N)
    pub n_fields: u64,
    /// 실현 최댓값 분포. 여기서 N_eff 를 역산합니다(Phase 2).
    pub realized_max: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpatialStat {
    /// 활성 패치 / 전체 패치
    pub active_ratio: Welford,
    /// 크롭 밖으로 밀려난 활성 패치 비율
    pub coverage_loss: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeStat {
    /// 축별 베이스라인 (TITLE FLOOR, 잡음대, pooled σ 등)
    #[serde(default)]
    pub baseline: HashMap<String, Welford>,
    /// 축별 감쇠 형상
    #[serde(default)]
    pub decay: HashMap<String, DecayStat>,
    /// 축별 후보 분산 (역분산 융합의 입력, Phase 1)
    #[serde(default)]
    pub axis_variance: HashMap<String, Welford>,
    /// 필드별 거절/확정 트레이스
    #[serde(default)]
    pub field: HashMap<String, FieldRejectStat>,
    /// 필드 쌍 혼동 사전
    #[serde(default)]
    pub confusion: HashMap<String, ConfusionStat>,
    /// 카테고리별 실현 최댓값
    #[serde(default)]
    pub category: HashMap<String, CategoryStat>,
    /// 비전 공간 통계
    #[serde(default)]
    pub spatial: HashMap<String, SpatialStat>,
    /// analytic 도메인 전이 카운트 ("from>to" → 횟수)
    #[serde(default)]
    pub transition: HashMap<String, u64>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SdsFile {
    pub recipe: String,
    pub team: String,
    pub updated_at: i64,
    #[serde(default)]
    pub scopes: HashMap<String, ScopeStat>,
}

static SDS: Lazy<RwLock<SdsFile>> = Lazy::new(|| RwLock::new(SdsFile::default()));
static DIRTY: Lazy<RwLock<bool>> = Lazy::new(|| RwLock::new(false));

// =====================================================================
// 🌟 [영속화]
// ---------------------------------------------------------------------
//  ── 위치 ──
//   앱 데이터 디렉터리 하위. bias.json 과 나란히 두면 '정적 사전 옆의
//   동적 사전' 이라는 관계가 파일 배치로 드러납니다.
//
//  ── 형식 ──
//   phrase_cache 의 anchors.bin 과 달리 JSON 입니다.
//   이 파일은 사람이 열어 "왜 그 판정이 나왔는가" 를 검수해야 하므로
//   가독성이 성능보다 우선합니다.
//
//  ── 삭제 ──
//   reset_lancedb / delete_all_models 시 동반 삭제됩니다.
//   삭제 = 즉시 현행(적응 이전) 동작 복귀이며, 이것이 롤백 수단입니다.
// =====================================================================
fn sds_path() -> std::path::PathBuf {
    crate::utils::get_app_dir().join("score_dynamics.json")
}

pub fn load(team: &str) {
    let path = sds_path();
    let mut fresh = SdsFile {
        recipe: SDS_RECIPE.to_string(),
        team: team.to_string(),
        updated_at: chrono::Utc::now().timestamp_millis(),
        scopes: HashMap::new(),
    };
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(txt) => match serde_json::from_str::<SdsFile>(&txt) {
                Ok(f) => {
                    if f.recipe != SDS_RECIPE {
                        println!(
                            "[SDS] 🧹 레시피 세대 불일치 (저장 '{}' vs 현재 '{}'). 통계를 전량 폐기하고 새로 시작합니다.",
                            f.recipe, SDS_RECIPE
                        );
                    } else if !team.is_empty() && !f.team.is_empty() && f.team != team {
                        println!(
                            "[SDS] 🧹 팀 불일치 (저장 '{}' vs 현재 '{}'). 통계를 전량 폐기합니다. (스코프 격리 원칙)",
                            f.team, team
                        );
                    } else {
                        let n = f.scopes.len();
                        fresh = f;
                        fresh.team = team.to_string();
                        println!("[SDS] ✅ 점수 동역학 통계를 불러왔습니다. 스코프 {}개.", n);
                    }
                }
                Err(e) => println!("[SDS] ⚠️ 통계 파싱 실패({}). 새로 시작합니다.", e),
            },
            Err(e) => println!("[SDS] ⚠️ 통계 읽기 실패({}). 새로 시작합니다.", e),
        }
    } else {
        println!("[SDS] 🆕 저장된 통계가 없습니다. 냉간 시작합니다. (현행 판정 그대로)");
    }
    if let Ok(mut w) = SDS.write() { *w = fresh; }
    // 🌟 [SEED WRITE] 부팅 시점에 빈 파일을 즉시 만듭니다.
    //
    //  ── 왜 필요한가 ──
    //   기존에는 flush 만 파일을 썼고, flush 는 dirty 일 때만 동작했습니다.
    //   그래서 '파일이 없다' 가 다음 셋 중 무엇인지 구분할 수 없었습니다.
    //     ① lib.rs 배선을 안 했다  ② enter_scope 를 못 탔다  ③ 아직 태스크가 안 끝났다
    //   부팅 즉시 빈 파일을 쓰면 ①과 ②③이 즉시 갈립니다.
    //   파일이 있는데 scopes 가 비어 있으면 ②③, 파일 자체가 없으면 ①입니다.
    //
    //  ── 비용 ──
    //   수백 바이트 1회 쓰기입니다.
    if let Ok(mut d) = DIRTY.write() { *d = true; }
    flush();
    println!("[SDS] 📍 통계 파일 경로: {}", sds_path().display());
}

/// 팀 식별자가 확정된 뒤(로그인 등) 호출합니다.
/// 다른 팀의 통계였다면 폐기합니다. 기획 6-4 의 스코프 격리 정책입니다.
pub fn rebind_team(team: &str) {
    let need_reset = SDS
        .read()
        .ok()
        .map(|s| !s.team.is_empty() && s.team != team)
        .unwrap_or(false);
    if need_reset {
        println!("[SDS] 🔄 팀 전환 감지. 이전 팀 통계를 폐기하고 새 팀으로 재바인딩합니다.");
        purge();
        load(team);
    } else if let Ok(mut w) = SDS.write() {
        w.team = team.to_string();
    }
}

pub fn flush() {
    let dirty = DIRTY.read().map(|d| *d).unwrap_or(false);
    if !dirty {
        // 🌟 [진단] '쓸 것이 없어서 안 썼다' 를 명시합니다.
        //    이 줄이 없으면 '호출은 됐는데 아무 일도 안 일어난' 상황과
        //    '호출 자체가 안 된' 상황을 구분할 수 없습니다.
        println!("[SDS] ⏭️ 새 관측이 없어 저장을 건너뜁니다. (dirty=false)");
        return;
    }
    let snapshot = match SDS.read() { Ok(s) => s.clone(), Err(_) => return };
    let mut snapshot = snapshot;
    snapshot.updated_at = chrono::Utc::now().timestamp_millis();
    snapshot.recipe = SDS_RECIPE.to_string();

    // 🌟 [상한] 기획 6-1 의 2MB 상한. 초과 시 관측 수가 적고 오래된 스코프부터 절삭합니다.
    let mut txt = match serde_json::to_string_pretty(&snapshot) {
        Ok(t) => t,
        Err(_) => return,
    };
    const CAP_BYTES: usize = 2 * 1024 * 1024;
    if txt.len() > CAP_BYTES {
        let mut keys: Vec<(String, u64, i64)> = snapshot
            .scopes
            .iter()
            .map(|(k, v)| {
                let obs: u64 = v.baseline.values().map(|w| w.n).sum::<u64>()
                    + v.field.values().map(|f| f.seen).sum::<u64>();
                (k.clone(), obs, v.updated_at)
            })
            .collect();
        // 관측이 적고 오래된 순으로 정렬
        keys.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
        let mut trimmed = snapshot.clone();
        for (k, _, _) in keys {
            if txt.len() <= CAP_BYTES { break; }
            trimmed.scopes.remove(&k);
            txt = serde_json::to_string_pretty(&trimmed).unwrap_or(txt);
        }
        println!("[SDS] ✂️ 상한 초과로 저관측 스코프를 절삭했습니다. (잔존 {}개)", trimmed.scopes.len());
    }

    let path = sds_path();
    if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
    match std::fs::write(&path, txt.as_bytes()) {
        Ok(_) => {
            if let Ok(mut d) = DIRTY.write() { *d = false; }
            println!("[SDS] 💾 점수 동역학 통계를 저장했습니다. ({}바이트)", txt.len());
        }
        Err(e) => println!("[SDS] ⚠️ 통계 저장 실패: {}", e),
    }
}

pub fn purge() {
    if let Ok(mut w) = SDS.write() {
        *w = SdsFile {
            recipe: SDS_RECIPE.to_string(),
            team: w.team.clone(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            scopes: HashMap::new(),
        };
    }
    let _ = std::fs::remove_file(sds_path());
    if let Ok(mut d) = DIRTY.write() { *d = false; }
    println!("[SDS] 🗑️ 점수 동역학 통계를 전량 삭제했습니다. 판정은 즉시 현행 상수로 복귀합니다.");
}

fn with_scope_mut<F: FnOnce(&mut ScopeStat, usize)>(f: F) {
    // 🌟 [UNSCOPED FALLBACK] 스코프 유무와 무관하게 반드시 기록합니다.
    let (key, ring) = effective_scope_key();
    if let Ok(mut w) = SDS.write() {
        let e = w.scopes.entry(key).or_insert_with(ScopeStat::default);
        f(e, ring);
        e.updated_at = chrono::Utc::now().timestamp_millis();
    }
    if let Ok(mut d) = DIRTY.write() { *d = true; }
    // 🌟 [AUTO FLUSH] 관측이 일정량 쌓이면 태스크 종료를 기다리지 않고 기록합니다.
    bump_and_maybe_flush();
}

// 🌟 [AUTO FLUSH] 태스크 끝까지 가야만 파일이 생기는 구조를 없앱니다.
// ---------------------------------------------------------------------
//  ── 왜 필요한가 ──
//   기존에는 flush 지점이 태스크 말미 한 곳뿐이라
//     · 이미지 1장이 크롭 10개 × Qwen3.5 로 수 분이 걸리는 동안 파일이 없고
//     · 중간에 `?` 로 에러가 전파되거나 사용자가 취소하면 관측이 통째로 사라지며
//     · 무엇보다 '지금 동작 중인가' 를 확인할 방법이 없었습니다.
//
//  ── 임계치가 매직 상수 아닌가 ──
//   이 값은 판정에 쓰이지 않습니다. '몇 번마다 디스크에 쓸 것인가' 라는
//   I/O 정책이며, 틀려도 통계나 추출 결과가 달라지지 않습니다.
//   파일 크기가 수십 KB 수준이라 자주 써도 부담이 없습니다.
static WRITE_TICK: Lazy<RwLock<u32>> = Lazy::new(|| RwLock::new(0));
const AUTO_FLUSH_EVERY: u32 = 64;

fn bump_and_maybe_flush() {
    let should = {
        match WRITE_TICK.write() {
            Ok(mut t) => {
                *t = t.wrapping_add(1);
                *t % AUTO_FLUSH_EVERY == 0
            }
            Err(_) => false,
        }
    };
    if should { flush(); }
}

// =====================================================================
// 🌟 [SSR] 관측 기록 API
// ---------------------------------------------------------------------
//  전부 반환값이 없고 실패해도 조용히 무시합니다.
//  계측이 판정을 방해하면 안 되기 때문입니다.
// =====================================================================

/// 축 베이스라인. TITLE FLOOR, 잡음대, pooled σ, net 표준편차 등.
pub fn record_baseline(axis: &str, value: f32) {
    if !value.is_finite() { return; }
    with_scope_mut(|s, ring| {
        s.baseline
            .entry(axis.to_string())
            .or_insert_with(Welford::default)
            .push(value as f64, ring);
    });
}

/// 순위 감쇠 곡선. 후보 점수 배열 전체를 넘기면 형상만 압축해 저장합니다.
pub fn record_decay(axis: &str, scores: &[f32]) {
    let shape = match decay_shape(scores) { Some(s) => s, None => return };
    with_scope_mut(|s, ring| {
        let d = s.decay.entry(axis.to_string()).or_insert_with(DecayStat::default);
        d.margin.push(shape.margin, ring);
        d.top_gap_ratio.push(shape.top_gap_ratio, ring);
        d.tail_flatness.push(shape.tail_flatness, ring);
        d.entropy_norm.push(shape.entropy_norm, ring);
        d.positive_ratio.push(shape.positive_ratio, ring);
    });
    // 축 분산은 역분산 융합(Phase 1)의 직접 입력이므로 별도 축으로도 남깁니다.
    let var = {
        let v: Vec<f64> = scores.iter().filter(|x| x.is_finite()).map(|x| *x as f64).collect();
        if v.len() < 2 { 0.0 } else {
            let m = v.iter().sum::<f64>() / (v.len() as f64);
            v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((v.len() - 1) as f64)
        }
    };
    if var > 0.0 {
        with_scope_mut(|s, ring| {
            s.axis_variance
                .entry(axis.to_string())
                .or_insert_with(Welford::default)
                .push(var, ring);
        });
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GateKind {
    Format,
    Prejudice,
    Enum,
    SelfId,
}

/// 필드가 후보로 검토되었음을 기록합니다.
pub fn record_field_seen(field: &str) {
    with_scope_mut(|s, _| {
        s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default).seen += 1;
    });
}

/// 게이트 거절. 학습형 prejudice(Phase 2)의 유일한 입력입니다.
///
/// ⚠️ 이 신호가 중요한 이유: 게이트 거절률은 '판정 결과' 가 아니라
///    '판정과 독립된 형식·구조 사실' 입니다. 판정 결과를 학습하면
///    자기강화 피드백(R1)에 걸리지만, 거절률은 그 위험이 없습니다.
pub fn record_field_reject(field: &str, kind: GateKind) {
    with_scope_mut(|s, _| {
        let e = s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default);
        match kind {
            GateKind::Format => e.reject_format += 1,
            GateKind::Prejudice => e.reject_prejudice += 1,
            GateKind::Enum => e.reject_enum += 1,
            GateKind::SelfId => e.reject_self_id += 1,
        }
    });
}

/// 필드 확정. 마진 분포를 함께 남겨 적응형 마진(Phase 1)의 기준선으로 씁니다.
pub fn record_field_assigned(field: &str, margin: f32) {
    with_scope_mut(|s, ring| {
        let e = s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default);
        e.assigned += 1;
        if margin.is_finite() {
            e.assign_margin.push(margin as f64, ring);
        }
    });
}

pub fn record_near_miss(field: &str) {
    with_scope_mut(|s, _| {
        s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default).near_miss += 1;
    });
}

/// 혼동 쌍. 값은 저장하지 않고 필드명만 남깁니다(프라이버시 정책).
pub fn record_confusion(winner: &str, loser: &str, margin: f32) {
    if winner.is_empty() || loser.is_empty() || winner == loser { return; }
    // 키 순서를 사전순으로 고정해 (A,B) 와 (B,A) 가 같은 레코드를 쓰게 합니다.
    let (a, b, winner_is_a) = if winner <= loser {
        (winner.to_string(), loser.to_string(), true)
    } else {
        (loser.to_string(), winner.to_string(), false)
    };
    let key = format!("{}|{}", a, b);
    with_scope_mut(|s, ring| {
        let e = s.confusion.entry(key).or_insert_with(ConfusionStat::default);
        e.ties += 1;
        if winner_is_a { e.a_wins += 1; } else { e.b_wins += 1; }
        if margin.is_finite() { e.margin.push(margin as f64, ring); }
    });
}

/// 카테고리 실현 최댓값. CATEGORY-NEUTRAL 의 N_eff 캘리브레이션(Phase 2) 입력.
pub fn record_category_max(category: &str, n_fields: usize, realized_max: f32) {
    if !realized_max.is_finite() { return; }
    with_scope_mut(|s, ring| {
        let e = s.category.entry(category.to_string()).or_insert_with(CategoryStat::default);
        e.n_fields = n_fields as u64;
        e.realized_max.push(realized_max as f64, ring);
    });
}

/// 비전 히트맵 확산도. V-1 공간 잔차화(Phase 1)의 판정 기준선.
pub fn record_spatial(category: &str, active: usize, total: usize) {
    if total == 0 { return; }
    let ratio = active as f64 / total as f64;
    with_scope_mut(|s, ring| {
        s.spatial
            .entry(category.to_string())
            .or_insert_with(SpatialStat::default)
            .active_ratio
            .push(ratio, ring);
    });
}

/// 크롭 커버리지 손실률. V-2(Phase 2) 입력.
pub fn record_coverage_loss(category: &str, lost_ratio: f32) {
    if !lost_ratio.is_finite() { return; }
    with_scope_mut(|s, ring| {
        s.spatial
            .entry(category.to_string())
            .or_insert_with(SpatialStat::default)
            .coverage_loss
            .push(lost_ratio as f64, ring);
    });
}

/// analytic 도메인 전이. U-2(Phase 3) 입력.
pub fn record_transition(from: &str, to: &str) {
    if from.is_empty() || to.is_empty() { return; }
    let key = format!("{}>{}", from, to);
    with_scope_mut(|s, _| {
        *s.transition.entry(key).or_insert(0) += 1;
    });
}

// =====================================================================
// 🌟 [ASE] 적응 파라미터 조회 API
// ---------------------------------------------------------------------
//  ⚠️ Phase 0 에서는 이 함수들이 정의만 되고 판정 경로에서 호출되지 않습니다.
//     기획 8-2 의 3단 게이팅(섀도 → 부분 → 전면) 중 섀도 이전 단계이며,
//     이 커밋을 적용해도 추출 결과가 바이트 단위로 동일해야 합니다.
//
//  ── 폴백 3단 ──
//   2차 스코프 → 1차 스코프 → 전역 → None(현행 동작)
//   관측 수가 Track::min_obs() 미달이면 다음 단계로 내려가고,
//   전부 미달이면 None 을 돌려줍니다. 이것이 냉간 시작 원칙의 구현입니다.
// =====================================================================
fn resolve<T, F>(pick: F) -> Option<T>
where
    F: Fn(&ScopeStat) -> Option<(T, u64)>,
{
    let scope = current_scope()?;
    let track = current_track()?;
    let (m2, m1, mg) = track.min_obs();
    let store = SDS.read().ok()?;
    for (key, need) in [
        (scope.key_secondary(), m2),
        (scope.key_primary(), m1),
        (scope.key_global(), mg),
    ] {
        if need == 0 { continue; }
        if let Some(st) = store.scopes.get(&key) {
            if let Some((val, n)) = pick(st) {
                if n >= need { return Some(val); }
            }
        }
    }
    None
}

/// 적응형 베이스라인. (평균, 표준편차) 를 돌려줍니다.
/// TITLE FLOOR / 잡음대 / dedup_floor 를 대체할 때 씁니다(Phase 1).
pub fn adaptive_baseline(axis: &str) -> Option<(f32, f32)> {
    resolve(|st| {
        st.baseline
            .get(axis)
            .map(|w| ((w.mean as f32, w.sd() as f32), w.n))
    })
}

/// 축 신뢰도. 역분산 융합의 가중치로 씁니다(Phase 1).
/// 분산이 작을수록(변별력 없음) 낮은 값을 돌려줍니다.
pub fn axis_confidence(axis: &str) -> Option<f32> {
    resolve(|st| {
        st.axis_variance.get(axis).map(|w| {
            let v = w.mean.max(1e-9);
            ((1.0 / v) as f32, w.n)
        })
    })
}

/// 이 축의 통상 감쇠 형상. 적응형 마진 판정의 기준선입니다(Phase 1).
/// (평탄도 평균, 평탄도 표준편차, 마진 평균, 마진 표준편차)
pub fn decay_baseline(axis: &str) -> Option<(f32, f32, f32, f32)> {
    resolve(|st| {
        st.decay.get(axis).map(|d| {
            (
                (
                    d.tail_flatness.mean as f32,
                    d.tail_flatness.sd() as f32,
                    d.margin.mean as f32,
                    d.margin.sd() as f32,
                ),
                d.margin.n,
            )
        })
    })
}

/// 학습형 특이도. prejudice 뱅크가 빈 필드의 대체 페널티입니다(Phase 2).
/// 거절률이 높은 필드일수록 큰 값을 돌려줍니다.
pub fn learned_specificity(field: &str) -> Option<f32> {
    resolve(|st| {
        st.field.get(field).map(|f| {
            let rejected = f.reject_format + f.reject_prejudice + f.reject_enum + f.reject_self_id;
            let denom = f.seen.max(1) as f64;
            ((rejected as f64 / denom) as f32, f.seen)
        })
    })
}

/// 혼동 사전. 동률에서 역사 승자를 돌려줍니다(Phase 1).
/// 반환값은 (승자 필드명, 승률).
pub fn confusion_winner(a: &str, b: &str) -> Option<(String, f32)> {
    if a.is_empty() || b.is_empty() || a == b { return None; }
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    let key = format!("{}|{}", x, y);
    let x_owned = x.to_string();
    let y_owned = y.to_string();
    resolve(move |st| {
        st.confusion.get(&key).map(|c| {
            let total = c.ties.max(1) as f32;
            if c.a_wins >= c.b_wins {
                ((x_owned.clone(), c.a_wins as f32 / total), c.ties)
            } else {
                ((y_owned.clone(), c.b_wins as f32 / total), c.ties)
            }
        })
    })
}

/// 실효 드로잉 수. CATEGORY-NEUTRAL 의 N 을 대체합니다(Phase 2).
///
///  ── 역산 근거 ──
///   Gumbel 기대 최댓값은 대략 √(2 ln N) 이므로,
///   실현 최댓값 평균 m 에서 N_eff ≈ exp(m² / 2) 로 되돌립니다.
///   구조가 가정한 필드 수 N 을 넘지 않도록 상한을 둡니다.
pub fn effective_draws(category: &str) -> Option<f32> {
    resolve(|st| {
        st.category.get(category).map(|c| {
            let m = c.realized_max.mean.max(0.0);
            let n_eff = (m * m / 2.0).exp().max(1.0);
            let cap = c.n_fields.max(1) as f64;
            (n_eff.min(cap) as f32, c.realized_max.n)
        })
    })
}

/// 비전 확산도 드리프트. V-1 폴백 판정에 씁니다(Phase 1).
pub fn spatial_drift(category: &str) -> Option<f32> {
    resolve(|st| {
        st.spatial
            .get(category)
            .map(|s| (s.active_ratio.drift_z() as f32, s.active_ratio.n))
    })
}

/// 도메인 전이 확률. analytic 동적 사전(Phase 3).
pub fn transition_prior(from: &str, to: &str) -> Option<f32> {
    let scope = current_scope()?;
    let track = current_track()?;
    let (m2, m1, _) = track.min_obs();
    let store = SDS.read().ok()?;
    for (key, need) in [(scope.key_secondary(), m2), (scope.key_primary(), m1)] {
        let st = match store.scopes.get(&key) { Some(v) => v, None => continue };
        let total: u64 = st
            .transition
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{}>", from)))
            .map(|(_, v)| *v)
            .sum();
        if total < need { continue; }
        let hit = st.transition.get(&format!("{}>{}", from, to)).copied().unwrap_or(0);
        // 디리클레 평활: 관측되지 않은 전이도 0 이 되지 않게 합니다.
        let k = st.transition.len().max(1) as f64;
        return Some((((hit as f64) + 1.0) / ((total as f64) + k)) as f32);
    }
    None
}

// =====================================================================
// 🌟 [진단] 현재 축적 상태 한 줄 요약. 태스크 종료 시 로그로 남깁니다.
// =====================================================================
pub fn report() -> String {
    let store = match SDS.read() { Ok(s) => s, Err(_) => return "[SDS] (잠금 실패)".to_string() };
    let scopes = store.scopes.len();
    let baseline_obs: u64 = store.scopes.values().flat_map(|s| s.baseline.values()).map(|w| w.n).sum();
    let decay_obs: u64 = store.scopes.values().flat_map(|s| s.decay.values()).map(|d| d.margin.n).sum();
    let field_obs: u64 = store.scopes.values().flat_map(|s| s.field.values()).map(|f| f.seen).sum();
    let confusion_obs: u64 = store.scopes.values().flat_map(|s| s.confusion.values()).map(|c| c.ties).sum();
    format!(
        "[SDS REPORT] 스코프 {}개 | 베이스라인 관측 {} | 감쇠 관측 {} | 필드 관측 {} | 혼동 관측 {} | (Phase 0: 판정 미개입)",
        scopes, baseline_obs, decay_obs, field_obs, confusion_obs
    )
}