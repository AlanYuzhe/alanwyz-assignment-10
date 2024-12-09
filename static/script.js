document.getElementById("search-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append("image-query", document.getElementById("image-query").files[0]);
    formData.append("text-query", document.getElementById("text-query").value);
    formData.append("lam", parseFloat(document.getElementById("lam").value));
    formData.append("query-type", document.getElementById("query-type").value);

    const usePca = document.getElementById("use-pca").checked;
    const pcaK = parseInt(document.getElementById("pca-k").value);

    formData.append("use-pca", usePca);
    formData.append("pca-k", isNaN(pcaK) ? 50 : pcaK);

    try {
        const response = await fetch("/search", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        const resultsDiv = document.getElementById("results");
        const resultImagesDiv = document.getElementById("result-images");

        resultImagesDiv.innerHTML = "";
        resultsDiv.style.display = "block";

        data.results.forEach(result => {
            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.alignItems = "center";
            container.style.marginBottom = "10px";

            const img = document.createElement("img");
            img.src = result.image;
            img.alt = `Similarity: ${result.similarity.toFixed(2)}`;
            img.style.maxWidth = "200px";
            img.style.marginRight = "10px";

            const similarityText = document.createElement("p");
            similarityText.textContent = `Similarity: ${result.similarity.toFixed(2)}`;
            similarityText.style.margin = "0";

            container.appendChild(img);
            container.appendChild(similarityText);
            resultImagesDiv.appendChild(container);
        });
    } catch (error) {
        console.error("Error fetching search results:", error);
    }
});