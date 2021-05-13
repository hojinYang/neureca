// variables
let userName = null;
let state = 'SUCCESS';

// functions
function Message(arg) {
    this.text = arg.text;
    this.message_side = arg.message_side;

    this.draw = function (_this) {
        return function () {
            let $message;
            $message = $($('.message_template').clone().html());
            $message.addClass(_this.message_side).find('.text').html(_this.text);
            $('.messages').append($message);

            return setTimeout(function () {
                return $message.addClass('appeared');
            }, 0);
        };
    }(this);
    return this;
}

function getMessageText() {
    let $message_input;
    $message_input = $('.message_input');
    return $message_input.val();
}

function sendMessage(text, message_side) {
    let $messages, message;
    $('.message_input').val('');
    $messages = $('.messages');
    message = new Message({
        text: text,
        message_side: message_side
    });
    message.draw();
    $messages.animate({ scrollTop: $messages.prop('scrollHeight') }, 300);
}

function greet() {
    setTimeout(function () {
        return sendMessage("Neuerca💡 here!", 'left');
    }, 1000);

    setTimeout(function () {
        return sendMessage("Please enter your id.", 'left');
    }, 2000);
}

function onClickAsEnter(e) {
    if (e.keyCode === 13) {
        onSendButtonClicked()
    }
}

function setUserName(username) {

    if (username != null && username.replace(" ", "" !== "")) {
        setTimeout(function () {
            return sendMessage("Hello, user id" + username + "!", 'left');
        }, 1000);
        setTimeout(function () {
            return sendMessage("I'll give you spot-on restaurant you'll love.", 'left');
        }, 2000);

        return username;

    } else {
        setTimeout(function () {
            return sendMessage("Please enter appropriate id.", 'left');
        }, 1000);

        return null;
    }
}

function requestChat(messageText, url_pattern) {
    $.ajax({
        url: "http://0.0.0.0:8080/" + userName + '/' + url_pattern + '/' + messageText,
        type: "GET",
        dataType: "json",
        success: function (data) {
            intent = data['intent'];

            if (intent === 'ask_information') {
                setTimeout(function () {
                    return sendMessage(data['review'], 'left');
                }, 500);

                if (data['preference'] != null) {
                    setTimeout(function () {
                        return sendMessage("You seem to have a preference for " + data['preference'], 'left');
                    }, 1000);
                };
                return null;

            } else if (intent == 'ask_history') {
                setTimeout(function () {
                    return sendMessage(data['history'], 'left');
                }, 1000);
            } else if (intent == 'ask_recommendation') {
                setTimeout(function () {
                    return sendMessage(data['rec'], 'left');
                }, 1000);

            } else {
                setTimeout(function () {
                    return sendMessage(data['intent'], 'left');
                }, 1000);
                setTimeout(function () {
                    return sendMessage(data['preference'], 'left');
                }, 1500);
            }
        },

        error: function (request, status, error) {
            console.log(error);

            return sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'left');
        }
    });
}

function onSendButtonClicked() {
    let messageText = getMessageText();
    sendMessage(messageText, 'right');

    if (userName == null) {
        userName = setUserName(messageText);

    } else {
        if (messageText.includes('Hi')) {
            setTimeout(function () {
                return sendMessage("Hi, Neu-Rec-Ca here!", 'left');
            }, 1000);
        } else if (messageText.includes('Thanks')) {
            setTimeout(function () {
                return sendMessage("Anytime!", 'left');
            }, 1000);

        } else if (state.includes('REQUIRE')) {
            return requestChat(messageText, 'fill_slot');
        } else {
            return requestChat(messageText, 'request_chat');
        }
    }
}