const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const placeholderContent = document.getElementById("placeholderContent");

const resultContainer = document.getElementById("resultContainer");
const resultText = document.getElementById("resultText");
const resultBox = document.getElementById("resultBox");
const loadingState = document.getElementById("loadingState");
const initialState = document.getElementById("initialState");
const resetBtn = document.getElementById("resetBtn");
const confidenceBar = document.getElementById("confidenceBar"); // Thêm thanh độ tin cậy

// --- SỰ KIỆN ---
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("bg-blue-50", "border-blue-400");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("bg-blue-50", "border-blue-400"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("bg-blue-50", "border-blue-400");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

resetBtn.addEventListener("click", resetApp);

// --- HÀM XỬ LÝ ---
function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Vui lòng chọn file ảnh!");
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove("hidden");
        placeholderContent.classList.add("hidden");
        sendToAI(file); // Gọi hàm gửi tới Server
    };
    reader.readAsDataURL(file);
}

// Hàm gửi ảnh tới Python (Flask)
async function sendToAI(file) {
    // 1. UI: Chuyển sang trạng thái loading
    initialState.classList.add("hidden");
    resultContainer.classList.add("hidden");
    loadingState.classList.remove("hidden");
    loadingState.classList.add("flex");

    // 2. Tạo Form Data để gửi file
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Gửi request tới server Python đang chạy ở cổng 5000
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Lỗi kết nối Server");

        const data = await response.json();
        
        if (data.error) {
            alert("Lỗi từ AI: " + data.error);
            resetApp();
        } else {
            showResult(data.result, data.confidence);
        }

    } catch (error) {
        console.error(error);
        alert("Không thể kết nối với AI Server! Hãy chắc chắn bạn đã chạy 'python app.py'.");
        resetApp();
    }
}

function showResult(label, confidence) {
    loadingState.classList.add("hidden");
    loadingState.classList.remove("flex");
    resultContainer.classList.remove("hidden");

    // Format phần trăm độ tin cậy
    const percent = (confidence * 100).toFixed(1) + "%";
    
    // Cập nhật thanh độ tin cậy
    confidenceBar.classList.remove("hidden");
    confidenceBar.querySelector("div").style.width = percent;

    if (label === "CHÍN") {
        resultText.innerHTML = `Cà chua CHÍN <br><span class="text-lg font-normal text-gray-600">Độ tin cậy: ${percent}</span>`;
        resultText.className = "text-3xl font-bold text-red-600";
        resultBox.className = "p-6 rounded-xl border-2 mb-6 bg-red-50 border-red-200 shadow-sm transition-all";
        confidenceBar.querySelector("div").className = "bg-red-600 h-2.5 rounded-full";
    } else {
        resultText.innerHTML = `Cà chua XANH <br><span class="text-lg font-normal text-gray-600">Độ tin cậy: ${percent}</span>`;
        resultText.className = "text-3xl font-bold text-green-600";
        resultBox.className = "p-6 rounded-xl border-2 mb-6 bg-green-50 border-green-200 shadow-sm transition-all";
        confidenceBar.querySelector("div").className = "bg-green-600 h-2.5 rounded-full";
    }
}

function resetApp() {
    fileInput.value = "";
    previewImage.src = "";
    previewImage.classList.add("hidden");
    placeholderContent.classList.remove("hidden");
    resultContainer.classList.add("hidden");
    initialState.classList.remove("hidden");
}