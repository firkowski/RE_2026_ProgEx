# Recursive Estimation 2026 - student container.
#
# Runs your code inside Docker so CPU and RAM limits match the grading
# environment. Results are saved to results.npz in the mounted folder;
# the launcher (run.bat / run.sh) then plots them on your host machine.
#
# Usage (from inside the exercise/ folder):
#     Windows : run.bat
#     Mac/Linux: bash run.sh
#
# Manual:
#     docker build -t re-student .
#     docker run --rm --cpus=4 --memory=2g --memory-swap=2g \
#         -v "$PWD":/work -w /work re-student
#
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /work
CMD ["/entrypoint.sh"]
