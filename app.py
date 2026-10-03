import os
import ssl
import re
import streamlit as st
import whisper
import yt_dlp
from google import genai

# SSL 인증서 관련 오류 방지
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="AI 설교 요약 보고서", page_icon="📖", layout="wide")

st.title("📖 AI 설교 요약 보고서 시스템")
st.write("유튜브 설교 영상 링크를 입력하시면 대본 추출 후 상세한 요약 보고서를 생성합니다.")

# Streamlit Secrets에서 API Key 가져오기
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

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

# 1단계: 유튜브 자막/대본 우선 추출 시도
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

# 2단계: 자막 없을 시 오디오 직접 다운로드
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

# 3단계: Whisper 모델을 통한 음성 텍스트 변환
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

# 4단계: 동적 모델 감지 적용 Gemini 요약 함수
def summarize_sermon_gemini(transcript_text, api_key):
    clean_key = api_key.strip()
    client = genai.Client(api_key=clean_key)
    
    prompt = f"""
다음은 설교 영상의 대본입니다:

[설교 대본]
{transcript_text}

---

위 대본을 깊이 있게 분석하여 성도들이 읽기 쉬운 상세한 [설교 요약 보고서]를 작성해 주세요.
보고서는 아래 항목을 반드시 포함하여 마크다운 형식으로 구성해 주세요:

1. 📖 **설교 제목 및 핵심 주제**
2. 📌 **주요 본문 구절** (대본 언급 기준)
3. 💡 **상세 설교 대지** (서론, 본론, 결론 구분 및 예화/인물/핵심 메시지 포함)
4. 🙏 **적용 및 묵상 기도 제목** (3가지)
"""
    # 1. 내 API 키로 접근 가능한 실제 모델 목록 전체 조회
    valid_models = []
    try:
        for m in client.models.list():
            # generateContent를 지원하는 모델만 수집
            if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods:
                valid_models.append(m.name)
            elif not hasattr(m, 'supported_generation_methods'):
                valid_models.append(m.name)
    except Exception as e:
        st.error(f"모델 목록 조회 실패: {str(e)}")

    if not valid_models:
        st.error("현재 API 키로 이용 가능한 Gemini 모델을 찾을 수 없습니다. API 키 권한을 확인해 주세요.")
        return None

    # 2. 조회된 모델들 중 차례대로 호출 시도
    last_exception = None
    for model_name in valid_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_exception = e
            continue

    st.error(f"모든 모델 호출 실패 (조회된 모델: {valid_models}): {str(last_exception)}")
    return None

# Streamlit UI 구성
raw_url = st.text_input("유튜브 영상 URL", placeholder="https://www.youtube.com/watch?v=...")
col1, col2 = st.columns(2)
with col1:
    start_time = st.text_input("시작 시간 (hh:mm:ss 또는 mm:ss)", "00:00")
with col2:
    end_time = st.text_input("종료 시간 (hh:mm:ss 또는 mm:ss)", "10:00")

if st.button("🚀 처리 시작하기"):
    video_url = raw_url.strip()
    if not video_url or ("youtube.com" not in video_url and "youtu.be" not in video_url):
        st.warning("올바른 유튜브 영상 URL을 입력해 주세요.")
    elif not GEMINI_API_KEY:
        st.error("Streamlit Secrets에 GEMINI_API_KEY가 설정되지 않았습니다.")
    else:
        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time) if end_time != "00:00" else 999999
        
        transcript = None
        with st.spinner("1단계: 유튜브 대본/자막 추출 중..."):
            try:
                transcript = get_youtube_transcript(video_url, start_sec, end_sec)
            except Exception:
                pass
            
            if not transcript:
                st.info("공식/자동 자막이 없어 오디오 분석(Whisper)으로 전환합니다.")
                audio_file = download_audio_fallback(video_url)
                transcript = transcribe_audio(audio_file)
        
        st.subheader("📝 추출된 대본")
        st.text_area("전체 대본", transcript, height=200)
            
        with st.spinner("2단계: Gemini AI 요약 보고서 생성 중..."):
            summary = summarize_sermon_gemini(transcript, GEMINI_API_KEY)
            if summary:
                st.subheader("💡 AI 설교 요약 보고서")
                st.markdown(summary)
