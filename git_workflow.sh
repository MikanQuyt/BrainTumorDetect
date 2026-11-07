#!/usr/bin/env bash
# ==============================================================================
# Script tự động hoá quy trình Git cho bài tập BRISC 2025
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "Sử dụng: ./git_workflow.sh <REPO_URL>"
    echo "Ví dụ: ./git_workflow.sh https://github.com/your-org/brisc2025-project.git"
    exit 1
fi

REPO_URL=$1
REPO_NAME=$(basename -s .git "$REPO_URL")
WORK_DIR=$(pwd)

echo "=== 1. CLONE REPO VỀ MÁY ==="
git clone "$REPO_URL"
cd "$REPO_NAME"

echo "=== 2. CHUYỂN VÀO NHÁNH DEV ==="
git checkout dev || git checkout -b dev origin/dev

echo "=== 3. TẠO NHÁNH ten_minh_tuan1 ==="
git checkout -b ten_minh_tuan1

echo "=== 4. SAO CHÉP CÁC SCRIPT VÀ KẾT QUẢ TIỀN XỬ LÝ ==="
mkdir -p src output
cp "$WORK_DIR/audit_dataset.py" src/
cp "$WORK_DIR/preprocess_pipeline.py" src/
cp "$WORK_DIR/audit_report.json" output/
cp -r "$WORK_DIR/output"/* output/ 2>/dev/null || true

echo "=== 5. COMMIT THAY ĐỔI ==="
git add .
git commit -m "feat: audit BRISC 2025 dataset, apply CLAHE, extract bboxes, and split 70/15/15 patient-level"

echo "=== 6. CHUYỂN VỀ NHÁNH DEV VÀ GỘP NHÁNH ==="
git checkout dev
git merge ten_minh_tuan1

echo "=== 7. PUSH LÊN NHÁNH DEV TRÊN REMOTE ==="
git push origin dev

echo "=== HOÀN TẤT THÀNH CÔNG! ==="
