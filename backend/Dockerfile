# 베이스 이미지
FROM python:3.11.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# pip 업그레이드
RUN python -m pip install --upgrade pip

# requirements.txt 복사
COPY requirements.txt /app/

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . /app

# 컨테이너 포트
EXPOSE 8000

# 개발용 실행 명령
CMD ["uvicorn", "main:app", "--reload"]
