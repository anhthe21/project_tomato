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

// --- XỬ LÝ SỰ KIỆN ---

// Click để chọn ảnh
dropZone.addEventListener("click", () => fileInput.click());

// Hiệu ứng khi kéo thả
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("bg-blue-50", "border-blue-400");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("bg-blue-50", "border-blue-400");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("bg-blue-50", "border-blue-400");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

resetBtn.addEventListener("click", resetApp);

// --- CÁC HÀM CHỨC NĂNG ---

function handleFile(file) {
    // Kiểm tra có phải ảnh không
    if (!file.type.startsWith("image/")) {
        alert("Vui lòng chỉ chọn file hình ảnh!");
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        // Hiển thị ảnh preview
        previewImage.src = e.target.result;
        previewImage.classList.remove("hidden");
        placeholderContent.classList.add("hidden");
        
        // Bắt đầu quy trình phân loại
        startClassification(file);
    };
    reader.readAsDataURL(file);
}

function startClassification(file) {
    // 1. Reset trạng thái cũ
    initialState.classList.add("hidden");
    resultContainer.classList.add("hidden");
    
    // 2. Hiện loading
    loadingState.classList.remove("hidden");
    loadingState.classList.add("flex");

    // 3. Giả lập thời gian xử lý (1.5 giây) để tạo cảm giác "AI đang tính toán"
    setTimeout(() => {
        const result = classifyTomatoLogic(); // Gọi hàm logic
        showResult(result);
    }, 1500); 
}

// Hàm Logic giả lập (Sau này bạn thay model thật vào đây)
function classifyTomatoLogic() {
    return Math.random() > 0.5 ? "CHÍN" : "XANH";
}

function showResult(status) {
    // Tắt loading
    loadingState.classList.add("hidden");
    loadingState.classList.remove("flex");

    // Hiện kết quả
    resultContainer.classList.remove("hidden");
    
    // Cập nhật UI dựa trên kết quả
    if (status === "CHÍN") {
        resultText.textContent = "Cà chua CHÍN";
        resultText.className = "text-3xl font-bold text-red-600";
        resultBox.className = "p-6 rounded-xl border-2 mb-6 bg-red-50 border-red-200 shadow-sm";
    } else {
        resultText.textContent = "Cà chua XANH";
        resultText.className = "text-3xl font-bold text-green-600";
        resultBox.className = "p-6 rounded-xl border-2 mb-6 bg-green-50 border-green-200 shadow-sm";
    }
}

function resetApp() {
    fileInput.value = ""; // Reset input
    previewImage.src = "";
    previewImage.classList.add("hidden");
    placeholderContent.classList.remove("hidden");
    
    resultContainer.classList.add("hidden");
    initialState.classList.remove("hidden");
}