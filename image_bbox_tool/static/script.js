// script.js
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let img = new Image();
let boxes = [];
let isDrawing = false;
let startX, startY;

// 上传图片
document.getElementById('imageUpload').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('上传失败: ' + data.error);
                return;
            }
            img.src = '/uploads/' + data.filename;
            img.onload = function () {
                // 保存原始尺寸
                window.originalWidth = img.width;
                window.originalHeight = img.height;

                // 计算显示尺寸（最大 800x600）
                const maxWidth = 800;
                const maxHeight = 600;
                let displayWidth = img.width;
                let displayHeight = img.height;

                if (displayWidth > maxWidth) {
                    const ratio = maxWidth / displayWidth;
                    displayWidth = maxWidth;
                    displayHeight = Math.round(displayHeight * ratio);
                }
                if (displayHeight > maxHeight) {
                    const ratio = maxHeight / displayHeight;
                    displayHeight = Math.round(displayHeight * ratio);
                    displayWidth = Math.round(displayWidth * ratio);
                }

                // 设置 Canvas
                canvas.width = displayWidth;
                canvas.height = displayHeight;
                ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

                // 保存缩放比例
                window.scaleX = window.originalWidth / displayWidth;
                window.scaleY = window.originalHeight / displayHeight;

                boxes = [];
            };
        })
        .catch(err => {
            console.error('Upload error:', err);
            alert('上传失败，请检查网络或文件格式');
        });
    }
});

// 鼠标事件
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);

function startDraw(e) {
    if (!img.src) return;
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
}

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    // 重绘
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawAllBoxes();

    // 画当前框
    const w = currentX - startX;
    const h = currentY - startY;
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(startX, startY, w, h);
    ctx.setLineDash([]);

    const label = document.getElementById('labelInput').value;
    ctx.font = '14px Arial';
    ctx.fillStyle = 'red';
    ctx.fillText(label, startX, startY - 8);
}

function stopDraw(e) {
    if (!isDrawing) return;
    isDrawing = false;
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    const w = currentX - startX;
    const h = currentY - startY;
    const label = document.getElementById('labelInput').value;

    if (Math.abs(w) > 10 && Math.abs(h) > 10) {
        // 转换为原始图像坐标
        const x = Math.round(Math.min(startX, currentX) * window.scaleX);
        const y = Math.round(Math.min(startY, currentY) * window.scaleY);
        const width = Math.round(Math.abs(w) * window.scaleX);
        const height = Math.round(Math.abs(h) * window.scaleY);

        boxes.push({ label, x, y, w: width, h: height });
    }

    // 最终绘制
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawAllBoxes();
}

function drawAllBoxes() {
    boxes.forEach(box => {
        // 转换为显示坐标
        const dx = box.x / window.scaleX;
        const dy = box.y / window.scaleY;
        const dw = box.w / window.scaleX;
        const dh = box.h / window.scaleY;

        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(dx, dy, dw, dh);

        ctx.font = '14px Arial';
        ctx.fillStyle = 'white';
        ctx.fillRect(dx, dy - 20, ctx.measureText(box.label).width + 8, 20);
        ctx.fillStyle = 'red';
        ctx.fillText(box.label, dx + 4, dy - 8);
    });
}

function clearAll() {
    boxes = [];
    if (img.src) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
}

function exportJSON() {
    if (!img.src) {
        alert("请先上传图片！");
        return;
    }
    if (boxes.length === 0) {
        if (!confirm("没有标注框，确定要导出吗？")) return;
    }

    const filename = img.src.split('/').pop();
    const data = {
        filename: filename,
        original_width: window.originalWidth,
        original_height: window.originalHeight,
        boxes: boxes
    };

    // 显示 JSON（可选）
    const output = document.getElementById('json-output');
    output.style.display = 'block';
    output.textContent = JSON.stringify(data, null, 2);

    // 下载 JSON 文件
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `annotation_${filename.split('.')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
}