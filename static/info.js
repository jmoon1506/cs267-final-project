new Chart(document.getElementById("perf"), {
  type: 'bar',
  data: {
    labels: [0, 1, 2, 3, 4],
    datasets: [{ 
        data: [86,114,106,106,107],
        label: "Africa",
        borderColor: "#3e95cd",
        fill: false
      },
    ]
  },
  options: {
    responsive: true,
    title: {
      display: true,
      text: 'Performance'
    },
    legend: {
      display: false
    },
    tooltips: {
      enabled: false
    },
    scales: {
      yAxes: [{
        ticks: {
          beginAtZero: true,
        }
      }]
    }
  }
});