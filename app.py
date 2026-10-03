import os
import subprocess
import ssl
import sys
import whisper
import streamlit as st
import ollama

# macOS SSL 인증서 오류 방지
ssl._create_default_https_context = ssl._create_unverified_context

def download_and_cut_video(url, start_time, end_time, output_filename="cut_result.mp4"):
    if os.path.exists(output_filename):
        os.remove(output_filename)
        
    yt_dlp_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--download-sections", f"*{start_time}-{end_time}",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--extractor-args", "youtube:player_client=android,web",
        "--force-overwrites",
        "-o", output_filename,
        url
    ]
    
    try:
        subprocess.run(yt_dlp_cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        fallback_cmd = [
            sys.executable, "-m", "yt_dlp",
            "--download-sections", f"*{start_time}-{end_time}",
            "-f", "bestvideo+bestaudio/best",
            "--force-overwrites",
            "-o", output_filename,
            url
        ]
        try:
            subprocess.run(fallback_cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"구간 다운로드 중 오류 발생: {e}")
            return False

@st.cache_resource
def load_whisper_model():
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = whisper.load_model("small", device=device)
    return model, device

def transcribe_audio_with_timestamps(video_file="cut_result.mp4", text_output="transcript.txt"):
    model, device = load_whisper_model()
    
    transcribe_kwargs = {"language": "ko"}
    if device == "cpu":
        transcribe_kwargs["fp16"] = False
        
    result = model.transcribe(video_file, **transcribe_kwargs)
    
    timeline_lines = []
    for segment in result["segments"]:
        start_sec = int(segment["start"])
        m, s = divmod(start_sec, 60)
        h, m = divmod(m, 60)
        
        time_str = f"[{h:02d}:{m:02d}:{s:02d}]" if h > 0 else f"[{m:02d}:{s:02d}]"
        line = f"{time_str} {segment['text'].strip()}"
        timeline_lines.append(line)
    
    full_timeline_text = "\n".join(timeline_lines)
    
    with open(text_output, "w", encoding="utf-8") as f:
        f.write(full_timeline_text)
        
    return full_timeline_text

def summarize_sermon_ollama(transcript_text):
    """EXAONE 3.5 양식 이탈 방지형 상세 요약 함수"""
    
    system_instruction = """당신은 설교 대본을 바탕으로 정교한 보고서를 작성하는 전문 목회 보조 AI입니다.
[필수 준수 사항]
1. 반드시 아래 제공되는 [보고서 출력 양식]의 목차와 구조(제목, 핵심주제, 주요본문, 상세설교대지 1,2,3, 적용기도제목)를 정확하게 똑같이 유지하여 작성하세요.
2. 설교 대본에 등장하는 예화(0점 맞은 아들 비유)와 성경 인물(다윗, 엘리, 히스기야, 요시야)의 비교 내용을 생략하지 말고 상세하게 풀어서 기술하세요.
3. 요약문을 짧게 축약하지 말고 성도들이 묵상할 수 있도록 구체적인 내용으로 풍성하게 작성하세요."""

    user_prompt = f"""[분석할 설교 대본]:
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

    try:
        response = ollama.chat(
            model='exaone3.5',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.1,  # 임의 변경 방지를 위해 극도로 낮춤
                'num_predict': 2048  # 출력 길이를 크게 확장
            }
        )
        summary_result = response['message']['content']
    except Exception as e:
        summary_result = f"Ollama 요약 중 오류가 발생했습니다.\n오류 내용: {e}"

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_result)
        
    return summary_result

def main():
    st.set_page_config(page_title="유튜브 설교 구간 추출 및 무료 AI 요약기", layout="wide")
    st.title("📹 유튜브 설교 구간 추출 & 100% 무료 AI 요약기")
    
    st.markdown("유튜브 URL과 구간을 입력하면, **영상 추출**, **타임스탬프 대본 생성**, **무료 로컬 AI 핵심 요약**을 진행합니다.")

    with st.sidebar:
        st.header("⚙️ 엔진 상태")
        st.success("🟢 무료 Ollama AI 엔진 사용 중")

    url = st.text_input("유튜브 영상 URL", "https://www.youtube.com/watch?v=OXEcrCGqcI4")
    
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input("시작 시간 (hh:mm:ss)", "00:26:30")
    with col2:
        end_time = st.text_input("종료 시간 (hh:mm:ss)", "00:56:44")

    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""

    if st.button("🚀 처리 시작하기", type="primary"):
        with st.status("1/3단계: 유튜브 구간 안전 다운로드 중...", expanded=True) as status:
            success = download_and_cut_video(url, start_time, end_time, "cut_result.mp4")
            if success:
                status.update(label="1/3단계: 영상 다운로드 완료!", state="complete")
            else:
                status.update(label="1/3단계: 다운로드 실패", state="error")
                st.stop()
            
        with st.status("2/3단계: Whisper AI 음성 분석 및 대본 추출 중...", expanded=True) as status:
            transcript_text = transcribe_audio_with_timestamps("cut_result.mp4", "transcript.txt")
            st.session_state.transcript = transcript_text
            status.update(label="2/3단계: 대본 변환 완료!", state="complete")

        with st.status("3/3단계: 무료 Ollama AI 설교 핵심 요약 작성 중...", expanded=True) as status:
            summary_text = summarize_sermon_ollama(transcript_text)
            st.session_state.summary = summary_text
            st.session_state.processed = True
            status.update(label="3/3단계: 모든 요약 완료!", state="complete")
            
        st.rerun()

    # 결과 화면 출력
    if st.session_state.processed or (os.path.exists("summary.txt") and os.path.exists("cut_result.mp4")):
        if not st.session_state.summary and os.path.exists("summary.txt"):
            with open("summary.txt", "r", encoding="utf-8") as f:
                st.session_state.summary = f.read()
        if not st.session_state.transcript and os.path.exists("transcript.txt"):
            with open("transcript.txt", "r", encoding="utf-8") as f:
                st.session_state.transcript = f.read()

        st.balloons()
        st.success("✅ 설교 영상 추출 및 핵심 요약이 완료되었습니다!")
        
        tab1, tab2, tab3 = st.tabs(["📖 AI 설교 핵심 요약", "📌 전체 타임스탬프 대본", "🎬 추출 영상"])
        
        with tab1:
            st.subheader("💡 AI 설교 요약 보고서")
            st.markdown(st.session_state.summary)
            st.download_button(
                label="📥 요약문 (.txt) 다운로드",
                data=st.session_state.summary,
                file_name="sermon_summary.txt",
                mime="text/plain"
            )

        with tab2:
            st.subheader("📌 타임스탬프 포함 대본 전문")
            st.text_area("전체 대본", st.session_state.transcript, height=450)

        with tab3:
            st.subheader("🎬 잘라낸 설교 영상")
            st.video("cut_result.mp4")
            with open("cut_result.mp4", "rb") as f:
                st.download_button(
                    label="📥 영상 (.mp4) 다운로드",
                    data=f,
                    file_name="cut_sermon.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()
