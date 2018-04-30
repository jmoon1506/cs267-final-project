var paramObj = {
	b:	new Params( 16, 16, 48 ),
	i:	new Params( 50, 50, 500 ),
	a:	new Params( 32, 32, 140 )
}

function Controls() {
	this.controlsForm = document.getElementById("controls");
	this.newGame = document.getElementById("newgame");
	this.solve = document.getElementById("solve");
	this.ctrlElements = this.controlsForm.elements;
	this.auto_solve = options.autostart;
	console.log(options.proc);
	if (options.proc == 'distrib') {
		procType = 'distrib';
		document.getElementById("proc_serial").checked = false;
		document.getElementById("proc_shared").checked = false;
		document.getElementById("proc_distrib").checked = true;
	}

	this.controlsForm.onsubmit = function(e) {
		e.preventDefault();
		theControls.auto_solve = false;
		theControls.newGameButton();
	}

	this.newGame.onsubmit = function(e) {
		e.preventDefault();
		theControls.auto_solve = false;
		Srand.seed(Date.now());
		set_seed(Math.floor(Srand.random()*10000))
		
		theControls.newGameButton();
	}
	
	radioControl( "tsize", this.resizeTiles );	
	radioControl( "level", this.changeLevel );
	
	this.customform = document.getElementById("customform");
	this.rowform = document.getElementById("rows");
	this.colform = document.getElementById("columns");
	this.bombform = document.getElementById("bombs");
	this.bomberrormsg = document.getElementById("bomberrormsg");
	this.applyerrormsg = document.getElementById("bomberrormsg");
	
	this.rows = null;
	this.rowform.onblur = function(e) {
		theControls.rows = theControls.validateNum(theControls.rowform, 1, 99, "rowerror", true);
	}
	
	this.cols = null;
	this.colform.onblur = function(e) {
		theControls.cols = theControls.validateNum(theControls.colform, 1, 99, "colerror", true);
	}
	
	this.bombs = null;
	this.bombform.onblur = function(e) {
		var area = theControls.rows * theControls.cols;
		theControls.bomberrormsg.textContent = "Must be a number between 0 and " +
			(area ? area : "9999");
		theControls.bombs = theControls.validateBombs(true);
	}
	
	this.newGameButton();
}

function print_num(n) {
	console.log('Got this from Python: ' + n);
}

Controls.prototype = {
	validateNum: function(item, min, max, errorId, soft) {
		var value = item.value;
		if ( soft && item.value == "" ) var valid = true;
		else {
			valid = /^\d+$/.test(value)
			if ( valid ) {
				var num = 1 * value;
				if ( num < min || num > max ) valid = false;
			}
		}
		document.getElementById(errorId).setAttribute( "class", valid ? "noerror" : "error" );
		return valid ? num : null;
	},
	
	validateBombs: function(soft) {
		var area = theControls.rows * theControls.cols;
		theControls.bomberrormsg.textContent = "Must be a number between 0 and " +
			(area ? area : "9999");
		return theControls.validateNum(theControls.bombform, 0, 9999, "bomberror", soft);
	},
	
	newGameButton: function(redrawChart=true) {
		totalComputeTime = 0;
		if (redrawChart) {
			theChart.data.labels = [];
			theChart.data.datasets[0].data = [];
			theChart.data.datasets[1].data = [];
			theChart.data.datasets[2].data = [];
			theChart.update();
		}
		window.clearInterval(computeTimerObj);
		computeTimerObj = null;

		var els = this.ctrlElements;
		if (els.level.value == 'c') {
			this.rows = this.validateNum(theControls.rowform, 1, 99, "rowerror", false);
			this.cols = this.validateNum(theControls.colform, 1, 99, "colerror", false);
			if ( this.rows == null || this.cols == null ) {
				return;
			}
			
			this.bombs = this.validateBombs(false);
			if ( this.bombs == null ) return;
			
			paramObj.c = new Params( this.rows, this.cols, this.bombs );
		}
		var newp = paramObj[els.level.value];
	
		if ( theBoard.num == null || !equalParams(newp, theBoard.num) ) {
			theBoard.makeBoard(newp, els.tsize.value);
		}

		if (this.auto_solve)
			document.getElementById('solve_auto').firstChild.value = "Stop auto-solve";
		else
			document.getElementById('solve_auto').firstChild.value = "Start auto-solve";
		
		theBoard.newGame();
	},

	resizeTiles: function(e) {
		theBoard.tileSize = theControls.ctrlElements.tsize.value;
		theBoard.tableElt.setAttribute("class", "tiles-" + theControls.ctrlElements.tsize.value );

		theBoard.allTiles( refreshImage );
	},

	changeLevel: function(e) {
		var level = theControls.ctrlElements.level.value;
		if ( level == 'c' ) {
 			theControls.customform.setAttribute("class", "showcustom" );
		}
		else {
 			theControls.customform.setAttribute("class", "hidecustom" );
			// if ( theBoard.game != PLAYING ) theControls.newGameButton();
			theControls.auto_solve = false;
			theControls.newGameButton();
		}
	},

	request_solve: function() {
		// console.log("send " + gameId);
		computeTimer.innerHTML = 0;
		computeTime = 0;
		computeTimerObj = window.setInterval(function() {
			computeTime += 0.01;
			if (Date.now() % 8 === 0)
				computeTimer.innerHTML = computeTime.toFixed(2);
		}, 10);
		$.ajax({
			type: 'POST',
			url: '/api/solve_next',
			dataType: 'json',
			contentType: 'application/json; charset=utf-8',
			data: JSON.stringify({"board":theBoard.getTileArray(),"gameId":gameId,"procType":procType}),
			success: function(callback) {
				window.clearInterval(computeTimerObj);
				computeTimerObj = null;
				if (callback[7] != gameId || theBoard.game == OVER) return;
				if (turn > theChart.data.labels.length)
					theChart.data.labels.push(turn);
				turn++;
				
				totalComputeTime += computeTime;
/*				var elems = theChart.data.labels.length;
				if (elems > 10) {
					for (var i = 0; i < elems; i++) {
						if (i % 5 != 4 && i != 0)
							theChart.data.labels[i] = '';
					}
				}*/
				if (callback[6] == 0)
					theChart.data.datasets[0].data.push(computeTime);
				else if (callback[6] == 1)
					theChart.data.datasets[1].data.push(computeTime);
				else
					theChart.data.datasets[2].data.push(computeTime);
				theChart.update();
				theBoard.uncoverTile(callback);
				if (theControls.auto_solve === true) {
					theControls.request_solve();
				}
			},
			error: function(error) {
				window.clearInterval(computeTimerObj);
				computeTimerObj = null;
				console.log(error);
			}
		});
	}
}

function radioControl(name, routine) {
	buttons = document.getElementsByName( name );
	for ( var i=0 ; i < buttons.length ; i++ ) {
		buttons[i].onclick = routine
	}

}

function equalParams(a,b) {
	if (a.rows != b.rows) return false;
	if (a.cols != b.cols) return false;
	if (a.bombs != b.bombs) return false;
	return true;
}
