// Attach a submit handler to the form
$( "#searchForm" ).submit(function( event ) {
 
  // Stop form from submitting normally
  event.preventDefault();
 
  // Get some values from elements on the page:
  var $form = $( this ),
    keyword = $form.find("input[name='keyword']").val(),
    location = $form.find("input[name='location']").val(),
    url = $form.attr("action" );
 
  // Send the data using post
  var posting = $.post( url, { keyword: keyword, location: location }, dataType='json' );
 
  // Put the results in a div
  posting.done(function( data ) {
    var content = $( data );
    console.log(content)

    $('#results').empty()
    $.each(data, function(k, v) {
      html = (
        '<div class="row search-result">'+
          '<div class="col-xs-2 vcenter"><img src="'+v.image_url+'"></div>'+
          '<div class="col-xs-4 vcenter"><a href="'+v.url+'" target="_blank">'+v.name+'</a></div>' +
          '<div class="col-xs-5 vcenter">'
      );
      $.each(v.location[5], function(i, loc_val) {
        html += '<div class="address">'+loc_val+'</div>'
      });
      html += (
          '</div>'+
          '<div class="col-xs-1 vcenter">'+
            '<input type="number" id="'+k+'" class="rating rating-input" data-clearable="" '+
            'data-icon-lib="fa" data-active-icon="fa-star" data-inactive-icon="fa-star-o" '+
            'data-clearable-icon="fa-trash-o" />'+
          '</div>'+
        '</div>'
      );
      $('#results').append(html)
    });
    $('input.rating').rating({});
  });
});