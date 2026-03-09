import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

EXPERT_SYSTEM_MESSAGES = {
	"データ分析の専門家": (
		"あなたはデータ分析の専門家です。ユーザーの質問に対して、"
		"分析観点、仮説、指標、実行手順を分かりやすく整理して回答してください。"
	),
	"マーケティングの専門家": (
		"あなたはマーケティング戦略の専門家です。ユーザーの質問に対して、"
		"ターゲット、訴求、施策、効果測定の観点で実践的に回答してください。"
	),
}


def get_llm_response(input_text: str, expert_type: str) -> str:
	"""入力テキストと専門家種別を受け取り、LLMの回答を返す。"""
	system_message = EXPERT_SYSTEM_MESSAGES.get(
		expert_type,
		"あなたは丁寧で実務的なアシスタントです。",
	)

	prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_message),
			("human", "{input_text}"),
		]
	)

	model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
	chain = prompt | model | StrOutputParser()
	return chain.invoke({"input_text": input_text})


st.set_page_config(page_title="専門家LLMアシスタント", page_icon="AI", layout="centered")
st.title("専門家LLMアシスタント")

st.markdown(
	"""
このWebアプリでは、入力テキストをLangChain経由でLLMに渡し、回答を表示します。

操作方法:
1. ラジオボタンで専門家タイプを選択
2. 入力フォームに質問や相談内容を入力
3. 「送信」を押して回答を確認

※ 実行には `OPENAI_API_KEY` を環境変数または `.env` に設定してください。
"""
)

selected_expert = st.radio(
	"専門家タイプを選択してください",
	options=list(EXPERT_SYSTEM_MESSAGES.keys()),
)

with st.form("question_form"):
	user_text = st.text_area("入力テキスト", placeholder="例: 新商品の販売戦略を考えてください")
	submitted = st.form_submit_button("送信")

if submitted:
	if not user_text.strip():
		st.warning("入力テキストを入力してください。")
	elif not os.getenv("OPENAI_API_KEY"):
		st.error("`OPENAI_API_KEY` が設定されていません。`.env` か環境変数に設定してください。")
	else:
		with st.spinner("LLMに問い合わせ中..."):
			try:
				answer = get_llm_response(user_text, selected_expert)
				st.subheader("回答")
				st.write(answer)
			except Exception as exc:
				st.error(f"エラーが発生しました: {exc}")

