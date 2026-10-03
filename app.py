import os
import ssl
import re
import streamlit as st
import whisper
import yt_dlp
from google import genai

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="AI 설교 요약 보고서", page_icon="📖", layout="wide")

st.title("📖 AI 설교 요약 보고서 시스템")
st.write("유튜브 설교 영상 링크를 입력하시면 대본 추출 후 상세한 요약 보고서를 생성합니다.")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# 시간(hh:mm:ss 또는 mm:ss)을 초 단위로 변환
def time_to_seconds(t_str):
    try:
        parts = list(map(int, t_str.split(':')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return int(t_str)
    except:
        return 0

# 1. 유튜브 자막 우선 추출 시도 (가장 빠르고 실패율 0%)
def get_youtube_transcript(url, start_sec, end_sec):
    ydl_opts = {
        'skip_download': True,
        'writesub': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['ko', 'en'],
        'quiet': True,
        'nocheckcertificate': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subtitles = info.get('subtitles') or info.get('automatic_captions')
        
        if not subtitles:
            return None
        
        # 한국어 자막선택, 없으면 첫번째 자막
        sub_lang = 'ko' if 'ko' in subtitles else list(subtitles.keys())[0]
        sub_url = next((item['url'] for item in subtitles[sub_lang] if item.get('ext') == 'json3'), None)
        
        if not sub_url:
            return None
            
        import requests
        resp = requests.get(sub_url).json()
        
        transcript_text = ""
        for event in resp.get('events', []):
            start = event.get('tStartMs', 0) / 1000.0
            if 'segs' in event:
                text = "".join([seg.get('utf8', '') for seg in event['segs']]).strip()
                if text and text != '\n':
                    if start_sec <= start <= end_sec:
                        m, s = divmod(int(start), 60)
                        h, m = divmod(m, 60)
                        time_str = f"[{h:02d}:{m:02d}:{s:02d}]" if h > 0 else f"[{m:02d}:{s:02d}]"
                        transcript_text += f"{time_str} {text}\n"
        return transcript_text if transcript_text.strip() else None

# 2. 자막이 없을 경우 오디오 다운로드 후 Whisper 처리
def download_audio_fallback(url, output_filename="audio_temp"):
    output_mp3 = f"{output_filename}.mp3"
    if os.path.exists(output_mp3):
        os.remove(output_mp3)

    ydl_opts = {
        'format': 'ba/b',
        'outtmpl': output_filename,
        'force_overwrites': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'quiet': True,
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_mp3

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

# Streamlit UI
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
        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time) if end_time != "00:00" else 999999
        
        transcript = None
        with st.spinner("1단계: 유튜브 대본/자막 추출 중..."):
            try:
                transcript = get_youtube_transcript(video_url, start_sec, end_sec)
            except Exception as e:
                pass
            
            # 자막 추출 실패 시 오디오 직접 추출 및 Whisper 실행
            if not transcript:
                st.info("공식/자동 자막이 없어 오디오 분석(Whisper)으로 전환합니다.")
                audio_file = download_audio_fallback(video_url)
                transcript = transcribe_audio(audio_file)
        
        st.subheader("📝 추출된 대본")
        st.text_area("전체 대본", transcript, height=200)
            
        with st.spinner("2단계: Gemini AI 요약 보고서 생성 중..."):
            summary = summarize_sermon_gemini(transcript, GEMINI_API_KEY)
            st.subheader("💡 AI 설교 요약 보고서")
            st.markdown(summary)
