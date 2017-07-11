let get_input_text = function() {
    let text = $("textarea#input_text").val()
    return {'input_text': text}
}

let send_text_json = function(text) {
    $.ajax({
        url: '/predict',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_prediction(data);
        },
        data: JSON.stringify(text)
    })
}

let display_prediction = function(answer) {
    $("span#section").html(answer.section_name)
};


$(document).ready(function() {

    $("button#predict").click(function() {
        let text = get_input_text();
        send_text_json(text);
    })

})
