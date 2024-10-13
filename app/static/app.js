document.getElementById("uploadForm").onsubmit = async function(event) {
    event.preventDefault();

    let formData = new FormData();
    let fileField = document.querySelector("#file");

    formData.append("file", fileField.files[0]);

    const response = await fetch("/uploadfile/", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("result").textContent = "Кластер: " + result.cluster;
};