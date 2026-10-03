import os
import subprocess
import ssl
import sys
import whisper
import streamlit as st
from google import genai

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="AI 설교 요약 보고서", page_icon="📖", layout="wide")

st.title("📖 AI 설교 요약 보고서 시스템")
st.write("유튜브 설교 영상 링크를 입력하시면 대본 추출 후 상세한 요약 보고서를 생성합니다.")

# Streamlit Secrets에서 Gemini API 키 가져오기
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

def download_and_cut_video(url, start_time, end_time, output_filename="cut_result.mp4"):
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    # ffmpeg를 이용해 안정적으로 오디오 구획을 자르는 yt-dlp 옵션
    command = [
        "yt-dlp",
        "--download-sections", f"*{start_time}-{end_time}",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", output_filename,
        "--force-overwrites",
        url
    ]
    subprocess.run(command, check=True)
    return output_filename

def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename, fp16=False)
    
    transcript_text = ""
    for segment in result['segments']:
        start = int(segment['start'])
        m, s = divmod(start, 60)
        h, m = divmod(m, 60)
        time_str = f"[{h:02d}:{m:02d}:{s:02d}]" if h > 0 else f"[{m:02d}:{s:02d}]"
        transcript_text += f"{time_str} {segment['text']}\n"
        
    return transcript_text

def summarize_sermon_gemini(transcript_text, api_key):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
[설교 대본]:
{transcript_text}

---

위 대본을 바탕으로 아래의 [보고서 출력 양식] 그대로 상세한 설교 보고서를 작성해 주세요.

[보고서 출력 양식]
# 📖 [설교 요약] 심판을 넘어선 하나님의 자비와 긍휼

📌 **핵심 주제**
- 하나님은 우리의 완벽함을 요구하시는 것이 아니라, 심판 앞에서도 주님의 자비하심과 긍휼을 구하는 자를 기뻐하십니다.

📖 **주요 본문**
- 열왕기하 22:19~23:3 / 사무엘하 12:16, 22

💡 **상세 설교 대지**

1. **서론: 어린 시절의 추억과 은혜**
   - **0점 맞은 아들의 예화**: 시험에서 0점을 받아온 아들을 무조건 혼내기보다 작은 노력에도 기뻐해주신 부모님의 사랑처럼, 하나님은 불완전한 0점 짜리 인생이라도 주님을 의지하며 나아오는 마음을 기뻐하십니다.
   - **완벽주의 신앙의 함정**: 하나님을 엄격한 감독관으로 오해할 때 기쁨을 잃어버리지만, 은혜의 하나님을 알 때 참된 평안이 시작됩니다.

2. **본론: 심판을 대하는 세 가지 신앙의 자세 (인물 비교)**
   - **다윗 (자비를 구하는 믿음)**: 나단 선지자의 심판 경고 앞에서도 절망하거나 자포자기하지 않고 끝까지 하나님의 자비하심을 기대하며 금식하고 기도했습니다. 하나님은 그 은혜로 '솔로몬'을 허락하셨습니다.
   - **엘리와 히스기야 (체념과 방치의 위험)**: 하나님을 엄한 분으로만 여겨 "주님의 뜻대로 되기를 바란다"며 자포자기하고 방치했습니다. 이는 다음 세대의 영적 타락이라는 비극으로 이어졌습니다.
   - **요시야 왕 (회개와 영적 유산)**: 저주의 말씀 앞에서도 옷을 찢으며 즉각 회개했습니다. 개인의 구원에 안주하지 않고 온 백성을 모아 말씀을 나누고 유월절을 지켰습니다. 이 영적 개혁은 바벨론 포로기 속에서도 다니엘, 에스겔 같은 다음 세대를 세우는 토대가 되었습니다.

3. **결론: 예수 그리스도 안에 있는 진정한 충만함**
   - 세상의 인간관계나 환경은 100% 만족을 줄 수 없지만, 우리의 모든 율법적 저주는 십자가에서 예수님이 대신 담당하셨습니다.
   - 예수 그리스도의 긍휼과 사랑 안에 거할 때만 참된 만족과 행복을 누리게 됩니다.

🙏 **적용 및 묵상 기도 제목**
1. 불완전한 내 모습 그대로 자비의 하나님 보좌 앞에 담대히 나아가게 하소서.
2. 나 혼자만의 신앙에 안주하지 않고 요시야처럼 가정과 다음 세대를 세우는 영적 통로가 되게 하소서.
3. 세상 조건에 흔들리지 않고 십자가 예수 그리스도 안에서 진정한 기쁨과 충만함을 누리게 하소서.
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return response.text

# Streamlit UI 구성
video_url = st.text_input("유튜브 영상 URL", "https://www.youtube.com/watch?v=...")
col1, col2 = st.columns(2)
with col1:
    start_time = st.text_input("시작 시간 (hh:mm:ss 또는 mm:ss)", "00:00")
with col2:
    end_time = st.text_input("종료 시간 (hh:mm:ss 또는 mm:ss)", "10:00")

if st.button("🚀 처리 시작하기"):
    if not GEMINI_API_KEY:
        st.error("Streamlit Secrets에 GEMINI_API_KEY가 설정되지 않았습니다.")
    else:
        with st.spinner("1단계: 영상 다운로드 중..."):
            audio_file = download_and_cut_video(video_url, start_time, end_time)
        
        with st.spinner("2단계: Whisper로 대본 추출 중..."):
            transcript = transcribe_audio(audio_file)
            st.subheader("📝 추출된 대본")
            st.text_area("전체 대본", transcript, height=200)
            
        with st.spinner("3단계: Gemini AI 요약 보고서 생성 중..."):
            summary = summarize_sermon_gemini(transcript, GEMINI_API_KEY)
            st.subheader("💡 AI 설교 요약 보고서")
            st.markdown(summary)
