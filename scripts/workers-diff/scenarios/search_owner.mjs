// fix-search-answer-owner: 검색의 답 대화 행(id = 문맥 id · from = 팀 · to = 작업 bcc · cc = 작업 cc · ref = task.ref)의 data 에 검색한 사람(by = task.from)을 남긴다.
// 열은 그대로다. 팀장의 검색(search_deepinfra 등 task.from = 0xowner)도 by 가 생겨 쓰기가 바뀐다 — 그 시나리오들에는 fixes 로 이 id 를 달았다.
export default ({ T0, REQUEST, ROWS, TASK, SALES, SEARCH_LLM }) => {
  // 팀원(0xmember)의 사이트 멤버 행(bcc = 작업 cc) — 그 사람의 범위에서 SALES 행을 찾는다(fix-search-member-scope)
  const member = { db: "commerce_logis_apac1_items", table: "items", values: {
    id: "M9", type: "member", from: "0xmember", to: "0xteam", cc: "0xowner", bcc: "shop.example.com", ref: "", data: { name: "lee" },
    created_at: T0 - 600000, updated_at: T0 - 600000,
  } };
  const LLM = SEARCH_LLM({
    context: { context: [{ language: "Korean", type: "goods", text: "무거운 제품" }] },
    contexts: { contexts: [{ type: "goods", text: "무거운 제품", substantial: "weight", find: "heavy", condition: { quantity: { lte: 100 } } }] },
  });
  return [
    // 팀원의 검색: 답 행(문맥 id)과 작업 id 의 행 모두 data.by = 0xmember, from 은 그대로 팀
    { name: "search_answer_owner_member", fixes: ["fix-search-answer-owner"],
      rows: ROWS(TASK({ type: "prompt", from: "0xmember", body: "무거운 제품 보여줘" }), [...SALES(4), member]), request: REQUEST(), llm: LLM },
  ];
};
