var turn = 0;

var theChart = new Chart(document.getElementById("perf"), {
  type: 'bar',
  data: {
    labels: [],
    datasets: [{ 
        data: [],
        label: "Compute time",
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
      xAxes: [{
        ticks: {
          beginAtZero: true,
          maxTicksLimit: 12,
        },
        scaleLabel: {
          display: true,
          labelString: 'Turn'
        }
      }],
      yAxes: [{
        ticks: {
          beginAtZero: true,
        },
        scaleLabel: {
          display: true,
          labelString: 'Compute time'
        }
      }],
    }
  }
});