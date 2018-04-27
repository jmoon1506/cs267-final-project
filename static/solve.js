$('#solve_next').submit(function(e) {
    e.preventDefault();
    theControls.request_solve();
});

$('#solve_auto').submit(function(e) {
    e.preventDefault();
    theControls.auto_solve = !theControls.auto_solve;
    if (theControls.auto_solve) {
        this.firstChild.value = "Stop auto-solve";
        theControls.request_solve();
    }
    else {
        this.firstChild.value = "Start auto-solve";
    }
});

function seed_click(e) {
    var newseed = document.getElementById('newseed');
    if (newseed.style.visibility === 'hidden') {
        newseed.style.visibility = 'visible';
    } else {
        newseed.style.visibility = 'hidden';
    }
};

$('#newseed_form').submit(function(e) {
    e.preventDefault();
    seed = document.getElementById("newseed_input").value;
    // console.log(newseed === seed);
    // if (newseed == seed) return;
    // seed = newseed;
    document.getElementById("seed").innerHTML = seed;
    document.getElementById('newseed').style.visibility = 'hidden';
    theControls.auto_solve = false;
    theControls.newGameButton();
});

function set_seed(val) {
    seed = val;
    document.getElementById("newseed_input").value = seed;
    document.getElementById("seed").innerHTML = seed;
}