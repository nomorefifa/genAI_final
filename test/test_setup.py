import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✅ OpenAI API 키 로드 성공")
else:
    print("❌ OpenAI API 키가 없습니다")
    exit(1)

# 2. OpenAI 연결 테스트
try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10
    )
    print("✅ OpenAI API 연결 성공")
except Exception as e:
    print(f"❌ OpenAI API 연결 실패: {e}")
    exit(1)

# 3. PDF 파일 확인
import glob
pdf_files = glob.glob("data/*.pdf")
print(f"✅ PDF 파일 {len(pdf_files)}개 발견")
if len(pdf_files) != 9:
    print(f"⚠️  경고: PDF 파일이 9개가 아닙니다 (현재 {len(pdf_files)}개)")

# 4. 필수 라이브러리 확인
try:
    import chromadb
    import langgraph
    import gradio
    print("✅ 모든 필수 라이브러리 설치 완료")
except ImportError as e:
    print(f"❌ 라이브러리 누락: {e}")
    exit(1)

print("\n🎉 Phase 1 완료! 다음 단계로 진행하세요.")