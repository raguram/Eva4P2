function performFaceSwap(url){
	var fileInput1 = document.getElementById('faceSwapUpload1').files;
	var fileInput2 = document.getElementById('faceSwapUpload2').files;

	if(!fileInput1.length || !fileInput2.length){
		return alert('Please choose a file to upload first');
	}
	
	var file1 = fileInput1[0];
	var filename1 = file1.name;
	
	var file2 = fileInput2[0];
	var filename2 = file2.name;

	var formData = new FormData();
	formData.append(filename1, file1);
	formData.append(filename2, file2);
	
	console.log(url);
	
	$.ajax({
		async: true,
		method: 'POST',
		url: url,
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})
	.done(function (response) {		
		document.getElementById("faceSwapResult").src = "data:image/png;base64," + response;
	})
	.fail(function (error) {
		alert("There was an error while sending request for face swap"); 
		console.log(error);
	});
};