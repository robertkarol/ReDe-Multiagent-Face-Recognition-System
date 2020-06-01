let allFiles = []
let dropArea

$(document).ready(function () {
    dropArea = document.getElementById('drop-area')
    form = document.getElementById('register-form')
    ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false)
    })
    ;['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false)
    })

    ;['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false)
    })
    dropArea.addEventListener('drop', handleDrop, false)
    form.addEventListener('submit', handleSubmit);
});

function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
}

function highlight(e) {
    dropArea.classList.add('highlight')
}

function unhighlight(e) {
    dropArea.classList.remove('highlight')
}

function handleSubmit(e) {
    e.preventDefault()
    clearErrors()
    if (validateForm()) {
        uploadFiles()
    }
}

function handleDrop(e) {
    let dt = e.dataTransfer
    let files = dt.files
    handleFiles(files)
}

function handleFiles(files) {
    files = [...files]
    allFiles.push(...files)
    files.forEach(previewFile)
}

function uploadFiles() {
    files = allFiles
    let name = $('#name').val().toLowerCase().replace(" ", "_")
    let location = $('#location').val()
    var url = 'http://127.0.0.1:5000/register/' + location + '/' + name
    var formData = new FormData()
    files.forEach((file, index) => {
        formData.append('file' + index, file)
    })
    fetch(url, {
        method: 'POST',
        body: formData
    }).then(() => {
        clearErrors()
        alert("Upload successful")
    })
    .catch(() => {
        clearErrors()
        logError("Error registering")
    })
}

function logError(error) {
    $('<p>', {
        text: error,
    }).appendTo('#errors-log');
}

function clearErrors() {
    $('#errors-log').empty()
}

function validateForm() {
    let valid = true
    if (allFiles.length < 8) {
        logError("You should provide at least 8 face images!")
        valid = false
    }
    if (!$('#name').val()) {
        logError("Name field must be filled!")
        valid = false
    }
    return valid
}

function previewFile(file) {
    let reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onloadend = function () {
        $('<img>', {
            src: reader.result,
        }).appendTo('#gallery');
    }
}