$(document).ready(function() {
    $('#uploadForm').submit(function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: '/',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                $('#result').text('Prediction: ' + response.prediction);
            },
            error: function(xhr, status, error) {
                $('#result').text('Error: ' + xhr.responseJSON.error);
            }
        });
    });
});