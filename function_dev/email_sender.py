import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()

def send_email_with_pdf(to_email, pdf_path, subject="AI 브리핑 보고서", body="요청하신 요약 PDF 파일을 첨부합니다."):
    """
    이메일로 PDF 파일을 전송하는 함수.
    Args:
        to_email (str): 수신자 이메일 주소
        pdf_path (str): 첨부할 PDF 파일 경로
        subject (str): 메일 제목
        body (str): 메일 본문 내용
    """
    # 환경변수 또는 직접 입력
    from_email = os.getenv("GMAIL_USER")      # 보내는 이메일 주소
    app_password = os.getenv("GMAIL_APP_PW")  # Gmail 앱 비밀번호

    if not from_email or not app_password:
        raise ValueError("GMAIL_USER 및 GMAIL_APP_PW 환경변수를 설정해야 합니다.")

    # 이메일 메시지 설정
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    # PDF 첨부
    with open(pdf_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(pdf_path)
    msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)

    # 메일 전송
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(from_email, app_password)
            smtp.send_message(msg)
        print(f"✅ 메일 전송 완료: {to_email}")
    except Exception as e:
        print(f"❌ 메일 전송 실패: {e}")
