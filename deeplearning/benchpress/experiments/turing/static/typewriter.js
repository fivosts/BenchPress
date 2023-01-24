// Configuration Variables

var seperators = [";", ",",".", ":", "[", "]", "{", "}", "(", ")"];
var seperatorColor = "#950096";
var functionColor = "#6196cc";
var keywords = ["global", "ulong", "local", "kernel", "void", "var", "new", "for", "if", "else", "while", "break", "continue", "return", "class", "struct", "function", "goto", "static", "public", "private", "protected", "void"]; //...
var keywordColor = "#cc99cd"; // OK
var stringColor = "#7ec699"; // OK
var escapeCharacters = ["\\n", "\\t", "\\'", '\\"', "\\\\"];
var escapeCharacterColor = "violet";
var operators = ["&&", "||", ">=", "<=", "==", "!==", "!=", "===", "=", "+", "-", "!", "*", "/", "%", "|", "&", "~", "^", "<", ">", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|="];
var operatorColor = "#67cdcc";
var unaryOperators = ["++", "--"];
var unaryOperatorColor = "#67cdcc";
var numberColor = "#f08d49"
var booleans = ["true", "false", "True", "False"];
var booleanColor = "#f08d49";
var specialValues = ["null", "undefined"];
var specialValueColor = "#f08d49";
var primitiveTypes = ["const", "unsigned", "int", "char", "bool", "boolean", "float", "double", "short", "long", "byte"];
var primitiveTypeColor = "#cc99cd";
var otherTypes = ["String", "string"];
var otherTypeColor = "#cc99cd";
var functionDeclarationColor = "#f08d49"
var declarationParameterColor = "#f08d49";
var commentColor = "#999";
var singleLineComment = "//";
var multiLineComment = "/*";
var multiLineCommentEnd = "*/";
var newline = "\n";
var allowedNumCharsBefore = ["0x", "0"]; // etc
var allowedNumCharsAfter = ["f", "l"]; // etc

// To do list:
// 1. Fix number parsing so that it correctly rejects invalids









// Helper functions for parsing
function cat(arr) {
    for (var i = 0; i < arr.length; i++) {
        this.push(arr[i]);
    }
}
function addSpan (elem, text, color) {
    newSpan = document.createElement ("span");
    newSpan.style.color = color;
    elem.appendChild (newSpan);
    newSpan.innerHTML = text;
}
var isAlpha = function(ch){
    return /^[A-Z]$/i.test(ch);
}
var isNumber = function (num) {
    return /^[0-9]$/i.test(num);
}
var isAlphaNumeric = function(ch) {
    return isNumber(ch) || isAlpha(ch);
}
function htmlEscape(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}
function htmlUnescape(str){
    return str
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&amp;/g, '&');
}
function isValidOperand(ch) {
	if (isAlphaNumeric(ch) || ch === "_" || ch === "$" || ch === " " || ch === "\n") return true;
	return false;
}
// Function to check to see if typewriter should speed up and skip unimportant characters
function isTypewriterSkipChar(ch) {
    if (ch === " " || ch === "\n" || ch === "\t") return true;
    return false;
}
// Function that checks if something is a keyword. Returns -1 if it is not.
function isKeyword (str, index) {
    for (var i = 0; i < keywords.length; i++) {
        if (str.substr(index, keywords[i].length) === keywords[i] &&
            !isAlphaNumeric(str[index - 1]) && !isAlphaNumeric(str[index + keywords[i].length])) return keywords[i];
    }
    return -1;
}

function isSingleLineComment (str, index) {
    if (str.substr(index, singleLineComment.length) === singleLineComment) {
        var newlineIndex = str.indexOf(newline, index);
        if (newlineIndex !== -1) {
            return str.substring(index, newlineIndex);
        }
        else {
        	return str.substring(index, str.length);
        }
    }
    // Return -1 if not comment
    return -1;
}
function isMultiLineComment (str, index) {
    if (str.substr(index, multiLineComment.length) === multiLineComment) {
        var indexOfEnd = str.indexOf(multiLineCommentEnd, index + multiLineComment.length);
        if (indexOfEnd !== -1) {
            return str.substring(index, indexOfEnd + multiLineComment.length);
        }
    }
    return -1;
}
function isOperator (str, index) {
    for (var i = 0; i < operators.length; i++) {
        if (str.substr(index, operators[i].length) === operators[i] && isValidOperand(str[index - 1]) && isValidOperand(str[index + operators[i].length])) {
            return operators[i];
        }
    }
    return -1;
}
function isUnaryOperator (str, index) {
    for (var i = 0; i < unaryOperators.length; i++) {
        if (str.substr(index, unaryOperators[i].length) === unaryOperators[i] && ((!isValidOperand(str[index - 1]) && isValidOperand(str[index + unaryOperators[i].length])) || (isValidOperand(str[index - 1]) && !isValidOperand(str[index + unaryOperators[i].length])))) {
            return unaryOperators[i];
        }
    }
    return -1;
}
function isString (str, index) {
    if (str[index] === '"' || str[index] === "'") {
        var indexMatchingQuote = str.indexOf(str[index], index + 1);
        var matchingQuote = str[index];
        if (indexMatchingQuote === -1) {
            return [];
        }
        tempTokens = [];
        var it = index + 1;
        tempTokens.push({
            tokenName: str[index],
            tokenColor: stringColor
        });
        while (str[it] !== matchingQuote && it < str.length) {
            var escapeAdded = false;
            for (var charIdx = 0; charIdx < escapeCharacters.length; charIdx++) {
                if (str.substr(it, escapeCharacters[charIdx].length) === escapeCharacters[charIdx]) {
                    escapeAdded = true;
                    tempTokens.push({
                        tokenName: escapeCharacters[charIdx],
                        tokenColor: escapeCharacterColor
                    });
                    it += escapeCharacters[charIdx].length;
                    break;
                }
            }
            if (!escapeAdded) {
                tempTokens.push({
                    tokenName: str[it],
                    tokenColor: stringColor
                });
                it += 1;
            }

        }
        tempTokens.push({
            tokenName: str[index],
            tokenColor: stringColor
        });
        return tempTokens;
        // if (indexMatchingQuote !== -1) {
        //     return str.substring(index, indexMatchingQuote + 1);
        // }
    }
    return [];
}
function isSeperator (str, index) {
    for (var i = 0; i < seperators.length; i++) {
        if (str.substr(index, seperators[i].length) === seperators[i]) {
            return seperators[i];
        }
    }
    return -1;
}

function isBoolean (str, index) {
    for (var i = 0; i < booleans.length; i++) {
        if (str.substr(index, booleans[i].length) === booleans[i]) {
            return booleans[i];
        }
    }
    return -1;
}

function isPrimitiveType (str, index) {
    for (var i = 0; i < primitiveTypes.length; i++) {
        if (str.substr(index, primitiveTypes[i].length) === primitiveTypes[i] && !isAlphaNumeric(str[index - 1]) && !isAlphaNumeric(str[index + primitiveTypes[i].length])) {
            return primitiveTypes[i];
        }
    }
    return -1;
}

function isSpecial (str, index) {
    for (var i = 0; i < specialValues.length; i++) {
        if (str.substr(index, specialValues[i].length) === specialValues[i]) {
            return specialValues[i];
        }
    }
    return -1;
}
// Number recognition still needs fixing
function isANumber (str, index) {
    var position = index;
    var isValid = true;
    var beforeIndexStart = -1;
    var beforeIndexEnd = -1;
    var afterIndexStart = -1;
    var afterIndexEnd = -1;
    var numberStart = -1;
    var numberEnd = -1;
    while (isValid) {
        if (numberStart === -1 && isNumber(str[position])) {
            numberStart = position;
        }
        else {
            for (var i = 0; i < allowedNumCharsBefore.length; i++) {
                if (str.substr(position, allowedNumCharsBefore[i].length) === allowedNumCharsBefore[i]) {
                    if (beforeIndexStart !== -1) {
                        isValid = false;
                        break;
                    }
                    beforeIndexStart = position;
                    position += allowedNumCharsBefore[i].length;
                    beforeIndexEnd = position;
                    break;
                }
            }
            for (var i = 0; i < allowedNumCharsAfter.length; i++) {
                if (str.substr(position, allowedNumCharsAfter[i].length) === allowedNumCharsAfter[i]) {
                    if (afterIndexStart !== -1) {
                        isValid = false;
                        break;
                    }
                    afterIndexStart = position;
                    position += allowedNumCharsAfter[i].length;
                    afterIndexEnd = position;
                    break;
                }
            }
            // End of number
            if (!isNumber(str[position])) {
                if (numberStart === -1) {
                    isValid = false;
                }
                else {
                    numberEnd = position;
                }
                break;
            }
        }
        position += 1;
    }
    // This part needs more work
    if (!isValid) return -1;
    else {
        var start = numberStart;
        var end = numberEnd;
        if (beforeIndexStart !== -1 && beforeIndexStart < numberStart) {
            start = beforeIndexStart;
        }
        if (afterIndexStart !== -1 && afterIndexStart > numberEnd) {
            end = afterIndexEnd;
        }
        // Is invalid if other characters are to the right or left
        if (isValidOperand(str[numberStart - 1]) || isValidOperand(str[numberEnd + 1])) return -1
        return str.substring(start, end);
    }
}

function isFunction (str, index) {
    //Valid function name first character
    if (isAlpha(str[index]) || str[index] === "_" || str[index] === "$") {
        var indexOfSpace = str.indexOf(" ", index);
        var indexOfLeftParen = str.indexOf("(", index);
        var indexOfRightParen = str.indexOf(")", index);
        var indexOfNewline = str.indexOf("\n", index);
        if (indexOfLeftParen < indexOfNewline && indexOfRightParen < indexOfNewline
            && (!isAlphaNumeric(str[index - 1]) && str[index - 1] !== "_" && str[index - 1] !== "$")) {
            var endIndex = index + 1;
            while (isAlphaNumeric(str[endIndex]) || str[endIndex] === "_" || str[endIndex] === "$") {
                endIndex++;
            }
            if (str[endIndex] === "(" && endIndex === indexOfLeftParen) {
                return str.substring(index, endIndex);
            }
            var countSpaces = endIndex;
            while (str[countSpaces] === " ") {
                countSpaces++;
            }
            if (str[countSpaces] === "(" && countSpaces === indexOfLeftParen) {
                return str.substring(index, endIndex);
            }
        }
        return -1;
    }
    return -1;
}

/* spanObject = {
    token: "Tokenname",
    count: "Count",
    color: "color"
}; */
function parseString (str) {
    var tokens = [];
    var i = 0;
    // var whitespaceObject = {
    //     tokenName: " ",
    //     tokenColor: "white",
    //     tokenCount: 0
    // }
    var comment, multiComment, string, functionText, boolean, primative, special, operator, number, seperator, keyword, number;
    while (i < str.length) {
        if ((comment = isSingleLineComment(str, i)) !== -1) {
            tokens.push({
                tokenName: comment,
                tokenColor: commentColor,
                tokenCount: 1
            });
            i += comment.length;

        }
        else if ((multiComment = isMultiLineComment(str, i)) !== -1) {
            tokens.push({
                tokenName: multiComment,
                tokenColor: commentColor,
                tokenCount: 1
            });
            i += multiComment.length;
        }
        else if ((string = isString(str, i)) !== -1) {
            tokens.push({
                tokenName: string,
                tokenColor: stringColor,
                tokenCount: 1
            });
            i += string.length;
        }
        else if ((functionText = isFunction(str, i)) !== -1) {
            tokens.push({
                tokenName: functionText,
                tokenColor: functionColor,
                tokenCount: 1
            });
            i += functionText.length;
        }
        else if ((boolean = isBoolean(str, i)) !== -1) {
            tokens.push({
                tokenName: boolean,
                tokenColor: booleanColor,
                tokenCount: 1
            });
            i += boolean.length;
        }
        else if ((primitive = isPrimitiveType(str, i)) !== -1) {
            tokens.push({
                tokenName: primitive,
                tokenColor: primitiveTypeColor,
                tokenCount: 1
            });
            i += primitive.length;
        }
        else if ((special = isSpecial(str, i)) !== -1) {
            tokens.push({
                tokenName: special,
                tokenColor: specialValueColor,
                tokenCount: 1
            });
            i += special.length;
        }
        else if ((operator = isOperator(str, i)) !== -1) {
            tokens.push({
                tokenName: operator,
                tokenColor: operatorColor,
                tokenCount: 1
            });
            i += operator.length;
        }
        else if ((seperator = isSeperator(str, i)) !== -1) {
            tokens.push({
                tokenName: seperator,
                tokenColor: seperatorColor,
                tokenCount: 1
            });
            i += seperator.length;
        }
        else if ((keyword = isKeyword(str, i)) !== -1) {
            tokens.push({
                tokenName: keyword,
                tokenColor: keywordColor,
                tokenCount: 1
            });
            i += keyword.length;
        }
        else if ((number = isANumber(str, i)) !== -1) {
            tokens.push({
                tokenName: number.toString(),
                tokenColor: numberColor,
                tokenCOunt: 1
            });
            i += number.length;
        }
        else {
            var char = str[i];
            if (str[i] === " ") {
                char = "&nbsp";
            }
        	tokens.push({
        		tokenName: char,
        		tokenColor: "white",
        		tokenCount: 1
        	});

        }
    }
    return tokens;
}
    function setupTypewriter(documentObject) {
        var typewriter = {
            destinationDocumentObject: documentObject,
            typeSpeed: 1,
            isRunning: false,
            tokens: [],
            textColor: "white",
            syntaxHighlighting: true,
            charPosition: 0,
            isScrollLock: false,
            scrollLockInterval: -1,
            currentToken: 0,
            setSyntaxHighlighting: function (isSyntaxHighlightingOn) {
                this.syntaxHighlighting = isSyntaxHighlightingOn;
            },
            setNotSyntaxColor: function (color) {
                this.textColor = color;
            },
            setTypeSpeed: function (newSpeed) {
                if (newSpeed < 0 || newSpeed > 10000) return;
                this.typewriterSpeed = newSpeed;
            },
            stop: function () {
                this.isRunning = false;
            },
            parseString: function (str) {
                this.tokens = [];
                let i = 0;
                var comment, multiComment, string, functionText, boolean, primative, special, operator, number, seperator, keyword, number;
                while (i < str.length) {
                    if ((comment = isSingleLineComment(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: comment,
                            tokenColor: commentColor
                        });
                        i += comment.length;
                    }
                    else if ((multiComment = isMultiLineComment(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: multiComment,
                            tokenColor: commentColor
                        });
                        i += multiComment.length;
                    }
                    else if ((string = isString(str, i)).length !== 0) {
                        for (var i2 = 0; i2 < string.length; i2++) {
                            this.tokens.push(string[i2]);
                        }
                        // Add all the lengths of string
                        for (var it = 0; it < string.length; it++) {
                            i += string[it].tokenName.length;
                        }
                    }
                    else if ((functionText = isFunction(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: functionText,
                            tokenColor: functionColor
                        });
                        i += functionText.length;
                    }
                    else if ((boolean = isBoolean(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: boolean,
                            tokenColor: booleanColor
                        });
                        i += boolean.length;
                    }
                    else if ((primitive = isPrimitiveType(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: primitive,
                            tokenColor: primitiveTypeColor
                        });
                        i += primitive.length;
                    }
                    else if ((special = isSpecial(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: special,
                            tokenColor: specialValueColor
                        });
                        i += special.length;
                    }
                    else if ((operator = isOperator(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: operator,
                            tokenColor: operatorColor
                        });
                        i += operator.length;
                    }
                    else if ((unaryOperator = isUnaryOperator(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: unaryOperator,
                            tokenColor: unaryOperatorColor
                        });
                        i += unaryOperator.length;
                    }
                    else if ((seperator = isSeperator(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: seperator,
                            tokenColor: seperatorColor
                        });
                        i += seperator.length;
                    }
                    else if ((keyword = isKeyword(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: keyword,
                            tokenColor: keywordColor
                        });
                        i += keyword.length;
                    }
                    else if ((number = isANumber(str, i)) !== -1) {
                        this.tokens.push({
                            tokenName: number.toString(),
                            tokenColor: numberColor
                        });
                        i += number.length;
                    }
                    else {
                    	this.tokens.push({
                    		tokenName: str[i],
                    		tokenColor: "white"
                    	});
                    	i += 1;
                    }
                }
            },
            clearMessage: function () {
                this.tokens = [];
            },
            clearScreen: function () {
                this.destinationDocumentObject.innerHTML = "";
            },
            reset: function () {
                this.clearMessage();
                this.clearScreen();
            },
            parseHtml: function(html) {
                this.parseString(htmlUnescape(html.innerHTML));
            },
            typeWaitTime: function () {
                return Math.round(Math.random() * this.typeSpeed) + 50;
            },
            skipTypeWaitTime: function () {
                return Math.round(Math.random() * this.typeSpeed / 2);
            },
            typeChar: function() {
                if (this.currentToken < this.tokens.length) {
                    // Go to a new token
                    if (this.syntaxHighlighting) {
                        addSpan(this.destinationDocumentObject, this.tokens[this.currentToken].tokenName[this.charPosition], this.tokens[this.currentToken].tokenColor);
                    }
                    else {
                        addSpan(this.destinationDocumentObject, this.tokens[this.currentToken].tokenName[this.charPosition], this.textColor);
                    }
                }
                this.charPosition += 1;
                if (this.charPosition >= this.tokens[this.currentToken].tokenName.length) {
                    this.currentToken += 1;
                    this.charPosition = 0;
                }
                if (this.currentToken >= this.tokens.length) {
                    console.log("Reached end of token array");
                }
                else if (isTypewriterSkipChar(this.tokens[this.currentToken].tokenName[this.charPosition]) && this.isRunning) {
                    var that = this;
                    setTimeout(function(){that.typeChar()}, this.skipTypeWaitTime());
                }
                else if (this.isRunning) {
                    var that = this;
                    setTimeout(function(){that.typeChar()}, this.typeWaitTime());
                }
            },
            type: function () {
                if (this.tokens === []) return;
                this.isRunning = true;
                var that = this;
                setTimeout(function() {that.typeChar()}, this.typeWaitTime());
            },
            scrollLock: function (scrollLockOn) {
                var that = this;
                if (scrollLockOn && !this.isScrollLock) {
                    this.scrollLockInterval = setInterval(function() {that.destinationDocumentObject.scrollTop
                        that.destinationDocumentObject.scrollTop = that.destinationDocumentObject.scrollHeight;
                    }, 50);
                    this.isScrollLock = true;
                }
                else if (!scrollLockOn && this.isScrollLock) {
                    clearInterval(that.scrollLockInterval);
                    this.isScrollLock = false;
                    this.scrollLockInterval = -1;
                }
            }
        };
        return typewriter;
    }