const API_URL = window.location.origin;

// Función para mostrar respuestas
function showResponse(elementId, response) {
    const element = document.getElementById(elementId);
    if (typeof response === 'object') {
        element.textContent = JSON.stringify(response, null, 2);
    } else {
        element.textContent = response;
    }
}

// Generate Dataset
document.getElementById('generateForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const numSamples = document.getElementById('numSamples').value;
    
    try {
        const response = await fetch(`${API_URL}/generate?num_samples=${numSamples}`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Accept': 'application/json',
            }
        });
        const data = await response.json();
        showResponse('generateResponse', data);
    } catch (error) {
        showResponse('generateResponse', `Error: ${error.message}`);
    }
});

// Generate from JSON
document.getElementById('jsonForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = document.getElementById('jsonFile').files[0];
    const numSamples = document.getElementById('jsonSamples').value;
    const contextQuery = document.getElementById('contextQuery').value;
    
    if (!file) {
        showResponse('jsonResponse', 'Por favor seleccione un archivo JSON');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(
            `${API_URL}/ollama-data-generate-from-json?num_samples=${numSamples}&context_query=${encodeURIComponent(contextQuery)}`,
            {
                method: 'POST',
                body: formData
            }
        );
        
        if (response.headers.get('Content-Type').includes('text/csv')) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'generated_data.csv';
            a.click();
            showResponse('jsonResponse', 'CSV descargado exitosamente');
        } else {
            const data = await response.json();
            showResponse('jsonResponse', data);
        }
    } catch (error) {
        showResponse('jsonResponse', `Error: ${error.message}`);
    }
});

// Query MongoDB
document.getElementById('mongoForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const limit = document.getElementById('limit').value;
    const skip = document.getElementById('skip').value;
    const format = document.getElementById('format').value;
    const datasetId = document.getElementById('datasetId').value;
    
    let url = `${API_URL}/get-mongodb-data?limit=${limit}&skip=${skip}&format=${format}`;
    if (datasetId) {
        url += `&dataset_id=${datasetId}`;
    }
    
    try {
        const response = await fetch(url);
        
        if (format === 'csv') {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'mongodb_data.csv';
            a.click();
            showResponse('mongoResponse', 'CSV descargado exitosamente');
        } else {
            const data = await response.json();
            showResponse('mongoResponse', data);
        }
    } catch (error) {
        showResponse('mongoResponse', `Error: ${error.message}`);
    }
}); 