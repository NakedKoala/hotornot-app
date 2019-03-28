var el = x => document.getElementById(x);

function showPicker(inputId) { el('file-input').click(); }

function showPicked(input) {
    el('result-label').innerHTML = ``;
    el('upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('image-picked').src = e.target.result;
        el('image-picked').className = '';
    }
    reader.readAsDataURL(input.files[0]);
}

function analyze() {
    var uploadFiles = el('file-input').files;
    if (uploadFiles.length != 1) alert('Please select 1 file to analyze!');

    el('analyze-button').innerHTML = 'Analyzing...';

        var fileData = new FormData();
        fileData.append('img_file', uploadFiles[0]);

        $.ajax({
            type: "POST",
            enctype: 'multipart/form-data',
            url: "https://hotornot-app.be/predict",
            data: fileData,
            processData: false,
            contentType: false,
            cache: false,
            timeout: 600000,
            success: function (res) {
                var res = JSON.parse(res);
                if ('score' in res)
                    el('result-label').innerHTML = `Facial Beauty Score = ${res['score']}`;
                else
                    el('result-label').innerHTML = `Error: ${res['error']}`;
                el('analyze-button').innerHTML = 'Analyze';

            },
            error: function (e) {
                console.log(e)
            }
        });

}
