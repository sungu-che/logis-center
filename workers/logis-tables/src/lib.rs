//! logis-center Workers 에만 쓰는 것 (`logis-worker-tables`).
//!
//! 공용 층(JS 값 모형 · D1 · fetch · gzip · aws4fetch · ethers)은 premium-discount 의 `premium-discount-worker-common` 을
//! git 리비전으로 가져다 `logis_worker_common` 이라는 이름으로 쓴다(workers/Cargo.toml). 그 crate 로 옮기지 않은 것이 여기 있다:
//! - [`tables`] 원본 index.ts 의 지역 · 언어 표 (scripts/workers-gen/gen_tables.py 로 생성, 손으로 고치지 말 것)
//! - [`dirty_json`] `normalizeToJsonString` · `parseDirtyJson` (proxy 의 LLM 출력 고치기)
//!
//! 두 모듈은 예전 `workers/common` 의 것을 바이트 그대로 옮겼다. 그 안의 `crate::jsv` · `crate::platform` 이 그대로 풀리게 아래에서 다시 내보낸다.

pub(crate) use logis_worker_common::{jsv, platform};

pub mod dirty_json;
pub mod tables;
