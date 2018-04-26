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