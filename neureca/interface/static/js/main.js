

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

function onClickAsEnter(e) {
    if (e.keyCode === 13) {
        onSendButtonClicked()
    }
}


function requestChat(messageText) {
    $.ajax({
        url: "http://0.0.0.0:8080/" + 'request_chat/' + messageText,
        type: "GET",
        dataType: "json",
        success: function (data) {
            text = data['text'];
            for (let i = 0; i < text.length; i++) {
                setTimeout(function () {
                    return sendMessage(text[i], 'left');
                }, 500 * (i + 1));
            }
            return null

        },
        error: function (request, status, error) {
            console.log(error);
            return sendMessage('Sorry, failed to connect to server', 'left');
        }
    });
}

function startChat() {
    $.ajax({
        url: "http://0.0.0.0:8080/" + 'start_chat',
        type: "GET",
        dataType: "json",
        success: function (data) {
            text = data['text'];
            for (let i = 0; i < text.length; i++) {
                setTimeout(function () {
                    return sendMessage(text[i], 'left');
                }, 500 * (i + 1));
            }
            return null

        },
        error: function (request, status, error) {
            console.log(error);
            return sendMessage('Sorry, failed to connect to server', 'left');
        }
    });
}

function onSendButtonClicked() {
    let messageText = getMessageText();
    sendMessage(messageText, 'right');
    return requestChat(messageText)
}