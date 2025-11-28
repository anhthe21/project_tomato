const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const resultText = document.getElementById("resultText");

// NHẤN ĐỂ CHỌN ẢNH
dropZone.addEventListener("click", () => {
    fileInput.click();
});

// KÉO ẢNH VÀO
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("bg-blue-50");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("bg-blue-50");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("bg-blue-50");
    handleFile(e.dataTransfer.files[0]);
});

// CHỌN ẢNH TỪ MÁY
fileInput.addEventListener("change", () => {
    handleFile(fileInput.files[0]);
});

// HÀM XỬ LÝ FILE
function handleFile(file) {
    if (!file) return;
    const reader = new FileReader();

    reader.onload = function(e) {
        previewImage.src = e.target.result;
        classifyTomato(file); 
    };

    reader.readAsDataURL(file);
}

// HÀM PHÂN BIỆT XANH / CHÍN 
function classifyTomato(file) {  
    console.log(file.name);
    const result = Math.random() > 0.5 ? "Cà chua CHÍN" : "Cà chua XANH";

    resultText.textContent = result;
    resultText.classList.toggle("text-green-600", result.includes("XANH"));
    resultText.classList.toggle("text-red-600", result.includes("CHÍN"));
}