var theTimer;
var theCounter;
var theBoard;
var theControls;
var gameId = 1;
var seed = 7119;
var turn = 1;
var procType = 'serial';
var computeTimer = document.getElementById('computeTime');
var computeTimerObj = null;
var computeTime = 0;
var totalComputeTime = 0;

onload = init;

function init() {
	theTimer = new Timer("timer");
	theCounter = new Counter("counter");
	theBoard = new Board();

	set_seed(seed);

	theControls = new Controls();
	if (theControls.auto_solve)
		theControls.request_solve();
}

function Params(r,c,b) {
	this.rows = r;
	this.cols = c;
	this.bombs = b;
}

function getURLParameter(name, defaultVal) {
  return decodeURIComponent((new RegExp('[?|&]' + name + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search)||[,""])[1]
  	.replace(/\+/g, '%20'))||defaultVal
}

function Board() {
	this.num = null;
	
	this.tableElt = document.getElementById("grid");
	
	this.faceElt = document.getElementById("face");
	this.faceElt.onclick = function(e) {
		theBoard.newGame();
	}

	this.newGame = function() {
		gameId++;
		Srand.seed(seed);
		this.game = WAITING;
		this.allTiles( function(t) { t.reset() } );
		theBoard.setBombs( this.num.bombs );
		theCounter.setTo( this.num.bombs );
		theTimer.reset();
		this.setFace("neutral");
		document.getElementById("solve_auto").firstChild.disabled = false;
		document.getElementById("solve_next").firstChild.disabled = false;
		turn = 1;
	}
	
	this.endGame = function(win) {
		this.game = OVER;
		theTimer.stop();
		this.setFace( win ? "happy" : "dead" );
		document.getElementById("solve_auto").firstChild.disabled = true;
		document.getElementById("solve_next").firstChild.disabled = true;
		theControls.auto_solve = false;
		computeTimer.innerHTML = totalComputeTime.toFixed(2) + " total";
		window.clearInterval(computeTimerObj);
		computeTimerObj = null;
	}
	
	this.makeBoard = function(p, t) {
		this.num = p;
		this.tileSize = t;
		this.tableElt.setAttribute("class", "tiles-"+this.tileSize)
		this.board = []	//array for tiles
	
		//clear out the existing table
		var ch = this.tableElt.firstChild;
		while (ch) {
			this.tableElt.removeChild(ch);
			ch = this.tableElt.firstChild;
		}
		
		//build new the board
		for ( i=0 ; i < this.num.rows ; i++ ) {
			var newRow = []
			var newRowElt = document.createElement('tr');
		
			for ( j=0 ; j < this.num.cols ; j++ ) {
				var newTile = new Tile(i,j);
				newRow.push(newTile);
				newRowElt.appendChild(newTile.tdElt)
			}
		
			this.board.push(newRow);
			this.tableElt.appendChild(newRowElt);
		}
	};
	
	this.setBombs = function(k) {
		var n = this.num.rows * this.num.cols;
		assert(k<=n, "Too many bombs!");
		this.nonBombs = n - k;
		while ( n > 0 ) {
			n--;
			if ( Srand.random() < k/n ) {	//prob of bomb should be k/n, never true if k=0, always true if k=n
				var j = n % this.num.cols;
				var i = Math.floor(n / this.num.cols);
				this.board[i][j].bomb = true;
				k--;
			}
		}
	};

	this.getTile = function(i,j) {
		if ( i < 0 || i >= this.num.rows ) return null;
		if ( j < 0 || j >= this.num.cols ) return null;
		return this.board[i][j];
	};
	
	this.allTiles = function(iter) {
		for ( i=0 ; i < this.num.rows ; i++ ) {
			for ( j=0 ; j < this.num.cols ; j++ ) {
				iter(this.board[i][j]);
			}
		}
	};

	this.getTileArray = function() {
		var arr = [];
		for (var j = 0 ; j < this.num.rows; j++) {
			arr.push([]);
			for (var i = 0 ; i < this.num.cols; i++) {
				var tile = this.board[j][i];
				if ( tile.status < 0 )
					arr[j].push(tile.bombNeighbors);
				else
					arr[j].push(-1);
			}
		}
		return arr;
	}

	this.uncoverTile = function(pos) {
		// console.log(pos);
		var tile = this.board[pos[1]][pos[0]];
		tile.leftClick();
	}
	
	this.setFace = function(str) {
		this.faceElt.setAttribute("class", str);
	}
}

//values of Tile.status
var UNCOVERED = -1;
var COVERED = 0;
var FLAG = 1;
var QUESTION = 2;
var EXCLAMATION = 3;	//not a status per se but used for displaying hints

var WAITING = 0;
var PLAYING = 1;
var OVER = 2;

function Tile(i,j) {
	this.myRow = i;
	this.myCol = j;
	this.tdElt = document.createElement('td');
	this.bombNeighbors = 0;

	var self = this;
	this.tdElt.onclick = function(e) {
		self.leftClick();
	};
	this.tdElt.oncontextmenu = function(e) {
		self.rightClick();
		return false;
	};
	this.tdElt.onmouseover = function(e) {
		self.hover();
	};
	this.tdElt.onmouseout = function(e) {
		self.unhover();
	};

	this.reset();
}

Tile.prototype = {
	reset: function() {
		this.bomb = false
		this.status = COVERED//		this.bombNeighbors = -1;	//unrevealed--not used in javascript version
		this.setImage( addSize("covered-"), retString("") );
	},

	hover: function(evtObj) {
		if (theBoard.game == OVER) return false;
		if (this.status == COVERED)
			theBoard.setFace("worried");
		else
			theBoard.setFace("neutral");
	},

	unhover: function(evtObj) {
		if (theBoard.game == OVER) return false;
		theBoard.setFace("neutral");
	},
	
	rightClick: function(evtObj) {
		//do nothing if game over or already uncovered
		if ( theBoard.game == OVER || this.status == UNCOVERED ) return false;
		if ( this.status == COVERED ) {
			theCounter.decrement();
			this.status = FLAG;
			this.setImage( addSize("covered-"), iconHTML("flag") );
			return false;
		}
		if ( this.status == FLAG ) {	//there could be a setting to go straight back to covered w/o going through ?
			theCounter.increment();
			this.status = QUESTION;
			this.setImage( addSize("covered-"), retString("?") );
			return false;
		}
		if ( this.status == QUESTION ) {
			this.status = COVERED;
			this.setImage( addSize("covered-"), retString("") );
			return false;
		}
		assert(false, "Tile has invalid status: "+this.status);
	},
	
	leftClick: function(evtObj) {
		if ( theBoard.game == WAITING ) {
			theTimer.start();
			theBoard.game = PLAYING;
		}
		if ( theBoard.game == OVER || this.status == FLAG || this.status == UNCOVERED ) return;
		if ( this.bomb ) {	//oops, you lose
			this.status = UNCOVERED;
			theCounter.decrement();
			this.setImage( addSize("redsquare-"), iconHTML("bomb") );

			theBoard.allTiles( function(t) {
				if ( t.status == UNCOVERED ) return;
				if ( t.bomb ) {
					t.setImage( addSize("uncovered-"), iconHTML("bomb") );
				}
				else if (t.status == FLAG) {
					theCounter.increment();
					t.setImage( addSize("uncovered-"), iconHTML("bombx") );
				} 
			} );
			theBoard.endGame(false);	//you lose
		}
		else {
			theBoard.setFace("neutral");
			this.uncoverNonbomb();
		}
	},
	
	uncoverNonbomb: function() {
		var neighbors = [];
		var bombNeighbors = 0;
		var i = this.myRow;
		var j = this.myCol;
		addIn( theBoard.getTile( i-1 , j-1 ) );
		addIn( theBoard.getTile( i-1 , j   ) );
		addIn( theBoard.getTile( i-1 , j+1 ) );
		addIn( theBoard.getTile( i   , j-1 ) );
		addIn( theBoard.getTile( i   , j+1 ) );
		addIn( theBoard.getTile( i+1 , j-1 ) );
		addIn( theBoard.getTile( i+1 , j   ) );
		addIn( theBoard.getTile( i+1 , j+1 ) );
		
		this.status = UNCOVERED;
		
		if (bombNeighbors > 0) {
			this.setImage( addSize( "n" + bombNeighbors + " uncovered-" ), retString(""+bombNeighbors) );
		}
		else {
			this.setImage( addSize("uncovered-"), retString("") );
		}
		
		theBoard.nonBombs--;
		this.bombNeighbors = bombNeighbors;
		
		if (theBoard.nonBombs == 0 ) {
			theBoard.allTiles( function(t) {
				if ( t.status == UNCOVERED ) return;
				if ( t.status != FLAG ) theCounter.decrement();
				assert(t.bomb, "Player won, but there is a covered non-bomb");
				t.setImage( addSize("covered-"), iconHTML("flag") );
			} );
			theBoard.endGame(true);
			return;
		}
		
		if (bombNeighbors > 0) return;
		
		for ( i=0 ; i < neighbors.length ; i++ ) {
			var t = neighbors[i];
			if ( t.status != UNCOVERED ) t.uncoverNonbomb();
		}

		function addIn(x) {
			if (x) {
				neighbors.push(x);
				if (x.bomb) bombNeighbors++;
			}
		}
	},
	
	setImage: function(c,h) {
		this.myClass = c;
		this.myHTML = h;
		refreshImage(this);
	}
}

function refreshImage(t) {
	t.tdElt.setAttribute( "class", t.myClass(theBoard.tileSize) );
	t.tdElt.innerHTML = t.myHTML(theBoard.tileSize) ;
}

function addSize(str) {
	return function(size) { return str + size ; }
}

function retString(str) {
	return function(size) { return str ; }
}

function iconHTML(name) {
	return function(size) {
		return '<div class="' + name + '-' + size + '"><img src="graphics/' + name + '-' + 
			size + '.png" /></div>';
	}
}

function Counter(element) {
	if (element) this.myElement = document.getElementById(element);
	
	this.show = function() {
		if (this.myValue >= 0) var str = ("00"+Math.min(this.myValue,999)).slice(-3);
		else var str = "-"+("0"+Math.min(-this.myValue,99)).slice(-2);
		this.myElement.textContent = str	//for IE<9, use innerHTML
	}
	
	this.setTo = function(k) {
		this.myValue = k;
		this.show();
	}
	
	this.decrement = function() {
		this.myValue--;
		this.show();
	}
	
	this.increment = function() {
		this.myValue++;
		this.show();
	}
}

function Timer(element) {
	if (element) this.myElement = document.getElementById(element);
	var self = this;
	this.timerFn = function() { self.increment() };
	this.timerObj = null

	this.reset = function() {
		if (this.timerObj) this.stop();
		this.setTo(0);
	}
	
	this.start = function() {
		this.timerObj = window.setInterval(this.timerFn, 1000);
	}

	this.stop = function() {
		window.clearInterval(this.timerObj);
		this.timerObj = null;
	}
}

Timer.prototype = new Counter(null)

function assert(condition, message) {
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message;
    }
}
