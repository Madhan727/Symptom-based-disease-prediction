document.addEventListener('DOMContentLoaded', function () {
    const ctx = document.getElementById('severityChart');
    if (ctx) {
        fetch('/api/severity_history')
            .then(response => response.json())
            .then(data => {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            label: 'Symptom Severity',
                            data: data.values,
                            borderColor: '#0d6efd',
                            backgroundColor: 'rgba(13, 110, 253, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 6,
                            pointBackgroundColor: '#fff',
                            pointBorderColor: '#0d6efd',
                            pointBorderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 10,
                                grid: {
                                    display: false
                                }
                            },
                            x: {
                                grid: {
                                    display: false
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            });
    }
});

// Dynamic form interactions
function toggleCustomSymptom() {
    const select = document.getElementById('symptom_select');
    const customDiv = document.getElementById('custom_symptom_div');
    if (select.value === 'custom') {
        customDiv.classList.remove('d-none');
        customDiv.querySelector('input').setAttribute('required', 'true');
    } else {
        customDiv.classList.add('d-none');
        customDiv.querySelector('input').removeAttribute('required');
    }
}
