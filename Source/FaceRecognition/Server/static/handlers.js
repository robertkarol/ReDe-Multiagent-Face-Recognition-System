let allFiles = []
let dropArea

$(document).ready(function () {
    dropArea = $('#drop-area')
    form = $('#register-form')
    viewErrors = $('#view-errors')
    ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.on(eventName, preventDefaults)
    })
    ;['dragenter', 'dragover'].forEach(eventName => {
        dropArea.on(eventName, highlight)
    })

    ;['dragleave', 'drop'].forEach(eventName => {
        dropArea.on(eventName, unhighlight)
    })
    dropArea.on('drop', handleDrop)
    form.on('submit', handleSubmit)
    viewErrors.on('click', handleViewErrors)
});

function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
}

function highlight(e) {
    dropArea.addClass('highlight')
}

function unhighlight(e) {
    dropArea.removeClass('highlight')
}

function handleRemoveImage(element) {
    let id = $(element).parent().attr('id').match(/\d/g).join("")
    allFiles.splice(id, 1)
    $(element).parent().remove()
}

function handleViewErrors(e) {
    let element = e.target
    element.classList.toggle("active")
    let content = element.nextElementSibling;
    if (content.style.maxHeight) {
        content.style.maxHeight = null;
    } else {
        content.style.maxHeight = content.scrollHeight + "px";
    }
}

function handleSubmit(e) {
    e.preventDefault()
    clearErrors()
    if (validateForm()) {
        uploadFiles()
            .then(() => {
                alert("Upload successful")
                allFiles = []
                $('#gallery').empty()
            })
            .catch(error => {
                logError("Error registering: " + error)
            })
    }
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

function logError(error) {
    $('#view-errors').css('visibility', 'visible')
    $('<p>', {
        text: error,
        class: 'error-message'
    }).appendTo('#errors-log');
}

function clearErrors() {
    $('#view-errors').css('visibility', 'hidden')
    $('#errors-log').empty()
}

function uploadFiles() {
    files = allFiles
    let name = $('#name').val().toLowerCase().replace(" ", "_")
    let location = $('#location').val()
    let url = 'http://127.0.0.1:5000/register/' + location + '/' + name
    let formData = new FormData()
    files.forEach((file, index) => {
        formData.append('file' + index, file)
    })
    return fetch(url, {
        method: 'POST',
        body: formData
    })
}

function handleDrop(e) {
    let dataTransfer = e.originalEvent.dataTransfer
    let files = dataTransfer.files
    handleFiles(files)
}

function handleFiles(files) {
    files = [...files]
    allFiles.push(...files)
    files.forEach(previewFile)
}

function previewFile(file) {
    let reader = new FileReader()
    previewFile.index = previewFile.index || 0
    reader.readAsDataURL(file)
    reader.onloadend = function () {
        $('<div>', {
            class: 'gallery-image-container',
            id: 'container-' + previewFile.index
        }).appendTo('#gallery')
        $('<img>', {
            src: reader.result,
            class: 'gallery-image'
        }).appendTo('#container-' + previewFile.index)
        $('<button>', {
            class: 'remove-image',
            text: 'X',
            onclick: 'handleRemoveImage(this)'
        }).appendTo('#container-' + previewFile.index)
        previewFile.index++
    }
}
