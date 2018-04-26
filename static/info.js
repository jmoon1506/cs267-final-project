var turn = 1;

var theChart = new Chart(document.getElementById("perf"), {
  type: 'bar',
  data: {
    labels: [],
    datasets: [
      { 
        data: [],
        label: "Complete BP",
        backgroundColor: "#75B9BE",
        borderColor: "#3e95cd",
        fill: false
      },
      { 
        data: [],
        label: "Partial BP",
        backgroundColor: "#EE7674",
        borderColor: "#3e95cd",
        fill: false
      },
      { 
        data: [],
        label: "Reduction",
        backgroundColor: "#A6B1E1",
        borderColor: "#3e95cd",
        fill: false
      },
    ]
  },
  options: {
    responsive: true,
    title: {
      display: true,
      text: 'Performance',
    },
    legend: {
      display: true,
      position: 'bottom',
      labels: {
        boxWidth: 20,
      },
      // reverse: true,
    },
    tooltips: {
      enabled: true,
      mode: 'index',
      callbacks: {
        title: function(tooltipItem) {
          console.log(tooltipItem);
          return "Turn " + (tooltipItem[0].index+1);
        },
        label: function(tooltipItem) {
          return Number(tooltipItem.yLabel).toFixed(4) + " seconds";
        }
      }
    },
    scales: {
      xAxes: [{
        stacked: true,
        ticks: {
          beginAtZero: true,
        },
        scaleLabel: {
          display: true,
          labelString: 'Turn'
        }
      }],
      yAxes: [{
        stacked: true,
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

function hover_perf(e) {
  theChart.options.tooltips.enabled = true;
};

function unhover_perf(e) {
  theChart.options.tooltips.enabled = false;
};