use serde_json::{json, Value};
use tauri::Emitter;
use crate::store::{TradeDocument, VectorStore};
use crate::scheduler::{entity_bcc, entity_id, entity_index, entity_seed, normalize_entity_key};
use crate::scheduler::indexing::save_item;
use crate::utils::canonical::{
    is_relay_placeholder, ledger_delta, ledger_prior, relay_key_for_type, relay_ref_index,
    relay_type_family, LedgerPrior, LEDGER_PLACEHOLDER_DELTA, RELAY_LINK_KEYS,
};

pub type StatsDiff = std::collections::HashMap<String, (i64, i64, i64)>;

pub struct RelayEnv<'a> {
    pub store: &'a VectorStore,
    pub app_handle: &'a tauri::AppHandle,
    pub task_id: &'a str,
    pub team_id: &'a str,
    pub from: &'a str,
    pub cc: &'a str,
    pub ref_val: &'a str,
    pub search_mode: &'a str,
}

impl<'a> RelayEnv<'a> {
    fn emit(&self, msg: &str) {
        println!("{}", msg);
        let _ = self.app_handle.emit(
            "task-console-log",
            json!({ "task_id": self.task_id, "text": format!("{}\n", msg) }),
        );
    }
}

#[derive(Debug, Default, Clone)]
pub struct BridgeOutcome {
    pub referenced: bool,
    pub referrers: usize,
    pub linked: usize,
    pub drafted: usize,
    pub confirmed_foreign: usize,
}

struct ForwardRef {
    key: String,
    ftype: Option<String>,
    index: u32,
    title: Option<String>,
}

pub fn add_delta(stats: &mut StatsDiff, t: &str, d: (i64, i64, i64)) {
    if d == (0, 0, 0) || t.trim().is_empty() {
        return;
    }
    let e = stats.entry(t.to_string()).or_insert((0, 0, 0));
    e.0 += d.0;
    e.1 += d.1;
    e.2 += d.2;
}

fn scalar_text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

pub fn bind_tracking_ref(item: &mut Value, page_type: &str, team_id: &str, cc: &str) -> Option<u32> {
    if relay_type_family(page_type) != "order" {
        return None;
    }
    let tn = scalar_text(item.get("tracking_number"));
    if normalize_entity_key(&tn).is_empty() {
        return relay_ref_index(item.get("tracking"));
    }
    let idx = entity_index("tracking", team_id, &entity_seed(cc, &tn));
    if let Some(obj) = item.as_object_mut() {
        obj.insert("tracking".to_string(), json!(idx));
    }
    Some(idx)
}

pub fn goods_array_refs(item: &mut Value, team_id: &str, cc: &str) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    if let Some(arr) = item.get_mut("goods").and_then(|v| v.as_array_mut()) {
        for g in arr.iter_mut() {
            let raw = {
                let id = scalar_text(g.get("id"));
                if id.is_empty() { scalar_text(g.get("no")) } else { id }
            };
            if normalize_entity_key(&raw).is_empty() {
                continue;
            }
            let idx = entity_index("goods", team_id, &entity_seed(cc, &raw));
            if let Some(o) = g.as_object_mut() {
                o.insert("index".to_string(), json!(idx));
            }
            if !out.iter().any(|(_, i)| *i == idx) {
                out.push(("goods".to_string(), idx));
            }
        }
    }
    out
}

pub fn settle_relay_keys(item: &mut Value, page_type: &str) -> Vec<String> {
    let own = relay_type_family(page_type);
    let mut moved: Vec<String> = Vec::new();
    let obj = match item.as_object_mut() {
        Some(o) => o,
        None => return moved,
    };
    for key in RELAY_LINK_KEYS.iter() {
        if *key == own.as_str() {
            continue;
        }
        let raw = match obj.get(*key) {
            Some(Value::String(s)) => s.trim().to_string(),
            _ => continue,
        };
        obj.remove(*key);
        if raw.is_empty() || raw.eq_ignore_ascii_case("null") || raw.eq_ignore_ascii_case("n/a") {
            continue;
        }
        let companion = format!("{}_title", key);
        let companion_empty = obj
            .get(&companion)
            .and_then(|v| v.as_str())
            .map_or(true, |s| s.trim().is_empty());
        if companion_empty {
            obj.insert(companion, json!(raw));
        }
        moved.push(key.to_string());
    }
    moved
}

fn collect_forward_refs(item: &Value, page_type: &str) -> Vec<ForwardRef> {
    let own = relay_type_family(page_type);
    let mut out: Vec<ForwardRef> = Vec::new();
    for key in RELAY_LINK_KEYS.iter() {
        if *key == own.as_str() {
            continue;
        }
        let idx = match relay_ref_index(item.get(*key)) {
            Some(i) => i,
            None => continue,
        };
        let ftype = match *key {
            "goods" | "order" | "tracking" => Some(key.to_string()),
            _ => None,
        };
        let title = item
            .get(&format!("{}_title", key))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        out.push(ForwardRef { key: key.to_string(), ftype, index: idx, title });
    }
    out
}

pub async fn find_referrers(
    store: &VectorStore,
    key: &str,
    index: u32,
    types: &[&str],
    cap: usize,
) -> Vec<(String, Value)> {
    if index == 0 || types.is_empty() {
        return Vec::new();
    }
    let quoted: Vec<String> = types.iter().map(|t| format!("'{}'", t.replace('\'', "''"))).collect();
    let needle = crate::store::json_property_needle(key, &json!(index));
    let filter = format!("type IN ({}) AND data LIKE '%{}%'", quoted.join(", "), needle);
    let docs = match store.get_all_items("items", cap.max(1), 0, Some(filter)).await {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    docs.into_iter()
        .filter_map(|d| {
            let v: Value = serde_json::from_str(&d.json_data).ok()?;
            if is_relay_placeholder(&v) || relay_ref_index(v.get(key)) != Some(index) {
                return None;
            }
            Some((d.id, v))
        })
        .collect()
}

async fn find_tracking_by_number(store: &VectorStore, tn: &str, self_id: &str) -> Option<(String, u32)> {
    let needle = crate::store::json_property_needle("tracking_number", &json!(tn));
    let filter = format!("type IN ('tracking', 'receiving', 'shipping') AND data LIKE '%{}%'", needle);
    let docs = store.get_all_items("items", 4, 0, Some(filter)).await.ok()?;
    docs.into_iter().find_map(|d| {
        if d.id == self_id {
            return None;
        }
        let v: Value = serde_json::from_str(&d.json_data).ok()?;
        let idx = relay_ref_index(v.get("index"))?;
        Some((d.id, idx))
    })
}

async fn load_doc(store: &VectorStore, id: &str) -> (Option<TradeDocument>, Option<Value>) {
    let doc = store.get_item_by_id("items", id).await.ok().flatten();
    let json_val = doc.as_ref().and_then(|d| serde_json::from_str::<Value>(&d.json_data).ok());
    (doc, json_val)
}

pub async fn bridge_relays(
    env: &RelayEnv<'_>,
    page_type: &str,
    self_id: &str,
    self_index: u32,
    item: &mut Value,
    extra: &[(String, u32)],
    stats: &mut StatsDiff,
) -> BridgeOutcome {
    let mut out = BridgeOutcome::default();
    let own_family = relay_type_family(page_type);

    let settled = settle_relay_keys(item, page_type);
    if !settled.is_empty() {
        crate::utils::score_dynamics::record_baseline("commerce.relay_key_settled", settled.len() as f32);
        env.emit(&format!(
            "  🧷 [RELAY LEDGER / SETTLE] {} '{}' | 연결 키 {:?} 에 index 가 아닌 글자 값이 들어 있어 {{키}}_title 로 옮겼습니다. 연결 키에는 파이프라인이 만든 index 만 남깁니다. 저장 시 숫자 모양 글자는 수치로 굳어 다음 회차에 index 로 오인되기 때문입니다.",
            page_type, self_id, settled
        ));
    }

    let mut refs = collect_forward_refs(item, page_type);
    for (t, idx) in extra.iter() {
        if *idx == 0 || refs.iter().any(|r| r.index == *idx) {
            continue;
        }
        let key = relay_key_for_type(t).unwrap_or("").to_string();
        if key.is_empty() || key == own_family {
            continue;
        }
        refs.push(ForwardRef { key, ftype: Some(t.clone()), index: *idx, title: None });
    }

    for r in refs.iter() {
        let mut fid = entity_id(env.team_id, r.index);
        if fid == self_id {
            continue;
        }
        let (mut existing, mut existing_json) = load_doc(env.store, &fid).await;

        if existing.is_none() && r.ftype.as_deref() == Some("tracking") {
            let tn = crate::utils::hash::normalize_identifier(&scalar_text(item.get("tracking_number")));
            if !tn.is_empty() {
                if let Some((tid, t_index)) = find_tracking_by_number(env.store, &tn, self_id).await {
                    env.emit(&format!(
                        "  🔄 [RELAY LEDGER / SECONDARY KEY] {} → tracking | index 자리는 비어 있지만 tracking_number '{}' 로 기존 문서 '{}' (index={}) 를 찾았습니다. 그 문서의 index 로 다시 묶습니다.",
                        page_type, tn, tid, t_index
                    ));
                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("tracking".to_string(), json!(t_index));
                    }
                    fid = tid;
                    let loaded = load_doc(env.store, &fid).await;
                    existing = loaded.0;
                    existing_json = loaded.1;
                }
            }
        }

        let expect_type = r.ftype.clone().unwrap_or_else(|| r.key.clone());
        let found_type = existing
            .as_ref()
            .map(|d| d.r#type.clone())
            .filter(|t| !t.trim().is_empty())
            .or_else(|| {
                existing_json
                    .as_ref()
                    .and_then(|v| v.get("type").and_then(|x| x.as_str()).map(|s| s.to_string()))
            })
            .unwrap_or_default();
        if !found_type.is_empty() && relay_type_family(&found_type) != relay_type_family(&expect_type) {
            crate::utils::score_dynamics::record_baseline("commerce.relay_type_guard", 1.0);
            env.emit(&format!(
                "  🔀 [RELAY LEDGER / TYPE GUARD] {} → {} (index={}) 자리의 문서가 '{}' 타입입니다. index 는 타입을 해시에 포함하므로 이 충돌은 기존 데이터 오염입니다. 연결도 초안 생성도 하지 않습니다.",
                page_type, expect_type, r.index, found_type
            ));
            continue;
        }

        match ledger_prior(existing_json.as_ref()) {
            LedgerPrior::Absent => {
                let ftype = match r.ftype.as_deref() {
                    Some(t) => t.to_string(),
                    None => {
                        env.emit(&format!(
                            "  ⚪ [RELAY LEDGER / NO TYPE] {} → '{}' 키(index={}) 는 coupon·event 공용 축이라 가리키는 타입이 하나로 정해지지 않습니다. 초안을 만들지 않고 상대 원본이 들어올 때까지 연결 대기로 둡니다.",
                            page_type, r.key, r.index
                        ));
                        continue;
                    }
                };
                let label = r.title.clone().unwrap_or_else(|| r.index.to_string());
                let mut draft = json!({
                    "id": fid.clone(),
                    "type": ftype.clone(),
                    "index": r.index,
                    "updated_at": 0,
                    "mode": env.search_mode,
                    "text": format!("{} {}", ftype, label),
                });
                if let Some(obj) = draft.as_object_mut() {
                    if let Some(title) = r.title.as_ref() {
                        obj.insert("title".to_string(), json!(title));
                    }
                    if ftype == "tracking" {
                        let tn = crate::utils::hash::normalize_identifier(&scalar_text(item.get("tracking_number")));
                        if !tn.is_empty() {
                            obj.insert("tracking_number".to_string(), json!(tn.clone()));
                            obj.insert("text".to_string(), json!(format!("tracking {}", tn)));
                        }
                        if let Some(back) = relay_key_for_type(page_type) {
                            obj.insert(back.to_string(), json!(self_index));
                        }
                    }
                }
                let foreign_bcc = entity_bcc(&ftype, env.cc);
                save_item(
                    env.store, "items", &fid, &ftype, draft, None,
                    env.from, env.team_id, env.cc, &foreign_bcc, env.ref_val, None,
                ).await;
                add_delta(stats, &ftype, LEDGER_PLACEHOLDER_DELTA);
                out.drafted += 1;
                crate::utils::score_dynamics::record_baseline("commerce.relay_ledger_draft", 1.0);
                env.emit(&format!(
                    "  📝 [RELAY LEDGER / DRAFT] {} → {} '{}' (index={}) 가 아직 없어 자리 초안을 만듭니다. 이 문서가 들고 있는 상대 식별자로 index 를 재현했으므로, {} 원본이 들어오면 같은 id 로 착지해 초안이 해소됩니다.",
                    page_type, ftype, fid, r.index, ftype
                ));
            }
            LedgerPrior::Placeholder => {
                out.linked += 1;
                if let (Some(title), Some(mut ej), Some(doc)) = (r.title.as_ref(), existing_json.clone(), existing.as_ref()) {
                    let has_title = ej
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map_or(false, |s| !s.trim().is_empty());
                    if !has_title {
                        if let Some(obj) = ej.as_object_mut() {
                            obj.insert("title".to_string(), json!(title));
                            obj.insert("text".to_string(), json!(format!("{} {}", found_type, title)));
                        }
                        save_item(
                            env.store, "items", &fid, &found_type, ej, None,
                            &doc.from, &doc.to, &doc.cc, &doc.bcc, &doc.r#ref, None,
                        ).await;
                    }
                }
            }
            LedgerPrior::Draft => {
                out.linked += 1;
                if let (Some(mut ej), Some(doc)) = (existing_json.clone(), existing.as_ref()) {
                    if let Some(obj) = ej.as_object_mut() {
                        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                    }
                    save_item(
                        env.store, "items", &fid, &found_type, ej, None,
                        &doc.from, &doc.to, &doc.cc, &doc.bcc, &doc.r#ref, None,
                    ).await;
                    add_delta(stats, &found_type, ledger_delta(LedgerPrior::Draft, true));
                    out.confirmed_foreign += 1;
                    crate::utils::score_dynamics::record_baseline("commerce.relay_ledger_confirm", 1.0);
                    env.emit(&format!(
                        "  ✅ [RELAY LEDGER / CONFIRM] {} → {} '{}' (index={}) 는 목록에서만 수집된 draft 였고 이제 이 문서가 참조합니다. 참조가 성립했으므로 draft 에서 count 로 옮깁니다.",
                        page_type, found_type, fid, r.index
                    ));
                }
            }
            LedgerPrior::Confirmed => {
                out.linked += 1;
            }
        }
    }

    if let Some(key) = relay_key_for_type(page_type) {
        let related = crate::logic::related(page_type);
        let referrer_types: Vec<&str> = related
            .iter()
            .copied()
            .filter(|t| relay_type_family(t) != own_family)
            .collect();
        let referrers = find_referrers(env.store, key, self_index, &referrer_types, 32).await;
        out.referrers = referrers.len();
        out.referenced = !referrers.is_empty();
    }

    crate::utils::score_dynamics::record_baseline("commerce.relay_ledger_linked", out.linked as f32);
    env.emit(&format!(
        "  🔗 [RELAY LEDGER] {} '{}' (index={}) | 정방향 연결 {}건 · 자리 초안 생성 {}건 · 상대 draft→count {}건 | 역방향 참조 {}건",
        page_type, self_id, self_index, out.linked, out.drafted, out.confirmed_foreign, out.referrers
    ));
    out
}