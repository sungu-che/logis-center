// fix-search-member-scope: 팀장이 아닌 사람의 자연어 검색은 그 사람의 멤버 범위(사이트 cc · 페이지 bcc + 팀 공통)의 행만 — 벡터로 찾은 행 · 조건 SQL 의 행 모두.
// 같은 데이터에서 팀장은 지금과 같이 팀 전체. 가짜 Vectorize 에 sales 벡터를 미리 넣는다(scenario.vectors — mocks 의 seedVectors).
import { ethers } from "ethers";

const hashId = (text) => ethers.computeAddress(ethers.hashMessage(text)).toLowerCase();
const CC = hashId("example.com");
const CC_UP = CC.toUpperCase();
const OTHER_CC = hashId("other.com");
const HOME_CC = hashId("logis.center");
const BCC_A = hashId("goods" + CC_UP); // 페이지 A (goods 상세)
const BCC_A_LIST = hashId("goods" + CC); // 같은 type 의 목록 꼴
const BCC_O = hashId("goods" + OTHER_CC);
const BCC_H = hashId("goods" + HOME_CC);
const INDEX = "commerce-logis-apac1-sales";

export default ({ T0, REQUEST, ROWS, TASK, SEARCH_LLM }) => {
  const item = (id, values) => ({ db: "commerce_logis_apac1_items", table: "items", values: { id, from: "0xowner", to: "0xteam", ref: "", created_at: T0 - 600000, updated_at: T0 - 600000, ...values } });
  const sale = (id, cc, bcc, i) => ({ db: "commerce_logis_apac1_sales", table: "sales", values: {
    id, type: "goods", from: "0xowner", to: "0xteam", cc, bcc, ref: "R" + id,
    data: { title: "상품 " + id, weight: 1000 + i * 300 }, created_at: T0 - 500000 + i * 1000, index: 3000 + i, status: 9, weight: 1000 + i * 300, quantity: i + 1,
  } });
  const SALES = [
    sale("SA1", CC, BCC_A, 0), sale("SA2", CC, BCC_A_LIST, 1), sale("SO1", OTHER_CC, BCC_O, 2), sale("SO2", OTHER_CC, BCC_O, 3), sale("SH1", HOME_CC, BCC_H, 4),
    // 이 사이트의 bcc 없는 행 — 사이트 멤버만 본다. 벡터가 여섯이라 가짜 Vectorize 의 기본 topK(5)를 넘는다: 팀으로 거른 상위 5 개에서
    // 범위 밖 행을 빼므로 범위 안 행이 상위 5 밖에 있으면 벡터 결과에서 빠진다(SQL 결과에는 남음)
    sale("SC1", CC, "", 5),
  ];
  const values = (seed) => Array.from({ length: 768 }, (_, i) => Math.round(Math.sin(seed * 7.1 + i * 0.37) * 1e6) / 1e6);
  const VECTORS = { [INDEX]: SALES.map((r, i) => ({ id: r.values.id, values: values(i + 1), metadata: { type: "goods", to: "0xteam", cc: r.values.cc, bcc: r.values.bcc } })) };
  const SCOPE_ROWS = (from, extra = []) => ROWS(TASK({ type: "prompt", from, cc: HOME_CC, body: "무거운 제품 보여줘" }), [
    ...SALES,
    item("PA", { type: "pages", cc: CC, bcc: BCC_A, data: { type: "goods", origin: "https://shop.example.com", link: "/goods/view", detail: true } }),
    item("M1", { type: "member", from: "0xmember", cc: "0xowner", bcc: BCC_A, ref: "PA", data: { name: "lee" } }),
    ...extra,
  ]);
  const LLM = SEARCH_LLM({
    context: { context: [{ language: "Korean", type: "goods", text: "무거운 제품" }] },
    contexts: { contexts: [{ type: "goods", text: "무거운 제품", substantial: "weight", find: "heavy", condition: { quantity: { lte: 100 } } }] },
  });
  return [
    // 페이지 A 의 멤버: A(목록 꼴 포함) · 팀 공통 행만 — 다른 사이트 행은 벡터 · SQL 모두에서 빠진다
    { name: "search_scope_member", fixes: ["fix-search-answer-owner"], rows: SCOPE_ROWS("0xmember"), vectors: VECTORS, request: REQUEST(), llm: LLM },
    // 팀장: 지금과 같이 팀 전체
    { name: "search_scope_owner", fixes: ["fix-search-answer-owner"], rows: SCOPE_ROWS("0xowner"), vectors: VECTORS, request: REQUEST(), llm: LLM },
    // 사이트 멤버(bcc = cc): 그 사이트 통째 + 팀 공통
    { name: "search_scope_site", fixes: ["fix-search-answer-owner"], rows: SCOPE_ROWS("0xsite", [item("M2", { type: "member", from: "0xsite", cc: "0xowner", bcc: CC, data: { name: "park" } })]),
      vectors: VECTORS, request: REQUEST(), llm: LLM },
  ];
};
