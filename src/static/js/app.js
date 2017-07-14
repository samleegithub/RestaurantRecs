var $searchLoading = $('#searchLoading').hide();
var $selectedRatingsGroup = $('#selectedRatingsGroup').hide();

// Attach a submit handler to the form
$( "#searchForm" ).submit(function( event ) {
 
  // Stop form from submitting normally
  event.preventDefault();
 
  // Get some values from elements on the page:
  var $form = $( this ),
    keyword = $form.find("input[name='keyword']").val(),
    location = $form.find("input[name='location']").val(),
    url = $form.attr("action" );
 
  $('#search-results').empty()
  $searchLoading.show()

  // Send the data using post
  var posting = $.post( url, { keyword: keyword, location: location }, dataType='json' );
 
  // Put the results in a div
  posting.done(function( data ) {
    var content = $( data );
    console.log(content)

    $searchLoading.hide()
    
    $.each(data, function(k, v) {
      html = (
        '<div class="row search-result-row">'+
          '<div class="col-xs-12 col-sm-2 vcenter cell results-image-div">'
      );
      if (typeof v.image_url != 'undefined' && v.image_url != '')
          html += '<img src="'+v.image_url+'" />'
      html += (
          '</div>' +
          '<div class="col-xs-12 col-sm-4 vcenter cell"><a href="'+v.url+'" class="text-faded" target="_blank">'+v.name+'</a></div>' +
          '<div class="col-xs-12 col-sm-4 vcenter cell">'
      );
      $.each(v.location[5], function(i, loc_val) {
        html += '<div class="address text-faded">'+loc_val+'</div>'
      });
      html += (
          '</div>'+
          '<div class="col-xs-12 col-sm-2 vcenter cell">'+
            '<input type="number" id="'+k+'" class="rating rating-input" data-clearable="" '+
            'data-icon-lib="fa" data-active-icon="fa-star" data-inactive-icon="fa-star-o" '+
            'data-clearable-icon="fa-trash-o" data-inline />'+
          '</div>'+
        '</div>'
      );
      $('#search-results').append(html)
    });
    $('input.rating').rating({});
  });
});