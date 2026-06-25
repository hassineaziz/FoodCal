document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('imageInput');
    const urlInput = document.getElementById('urlInput');
    const foodNameInput = document.getElementById('foodNameInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const removeBtn = document.getElementById('removeBtn');
    const form = document.querySelector('form');
    const submitBtn = document.querySelector('button[type="submit"]');

    // Handle click on upload area
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('drag-over');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('drag-over');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files[0]) {
            fileInput.files = files;
            handleFile(files[0]);
        }
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            previewImage.src = reader.result;
            previewContainer.classList.add('active');
            uploadArea.style.display = 'none';
            urlInput.value = ''; // Clear URL input
            foodNameInput.value = ''; // Clear food name input
            updateButtonText('Identify Food');
        }
    }

    // Handle URL input
    urlInput.addEventListener('input', (e) => {
        if (e.target.value) {
            fileInput.value = '';
            previewContainer.classList.remove('active');
            uploadArea.style.display = 'block';
            foodNameInput.value = ''; // Clear food name input
            updateButtonText('Identify Food');
        }
    });

    // Handle Food Name input
    foodNameInput.addEventListener('input', (e) => {
        if (e.target.value) {
            fileInput.value = '';
            previewContainer.classList.remove('active');
            uploadArea.style.display = 'block';
            urlInput.value = ''; // Clear URL input
            updateButtonText('Search Calories');
        } else {
            updateButtonText('Identify Food');
        }
    });

    function updateButtonText(text) {
        submitBtn.textContent = text;
    }

    // Handle remove button
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        previewContainer.classList.remove('active');
        uploadArea.style.display = 'block';
        previewImage.src = '';
        updateButtonText('Identify Food');
    });

    // Form submission validation
    form.addEventListener('submit', (e) => {
        const hasFile = fileInput.files.length > 0;
        const hasUrl = urlInput.value.trim().length > 0;
        const hasName = foodNameInput.value.trim().length > 0;

        if (!hasFile && !hasUrl && !hasName) {
            e.preventDefault();
            alert('Please provide an image, a URL, or a food name');
        }
    });
});
