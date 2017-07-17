var $searchLoading = $('#search-loading');
var $searchResults = $('#search-results');
var $selectedRatingsGroup = $('.selected-ratings-group');
var $selectedRatings = $('#selected-ratings');
var all_ratings = {}
var valid_rating_values = {1:true, 2:true, 3:true, 4:true, 5:true}

// Attach a submit handler to the form
$("#searchForm").submit(function(event) {
 
  // Stop form from submitting normally
  event.preventDefault();
 
  // Get some values from elements on the page:
  var $form = $(this);
  var keyword = $form.find("input[name='keyword']").val();
  var location = $form.find("input[name='location']").val();
  var url = $form.attr("action");
 
  $searchResults.empty()
  $searchLoading.show()

  // Send the data using post
  var posting = $.post(url, { keyword: keyword, location: location }, dataType='json');
 
  // Put the results in a div
  posting.done(function(data) {
    var content = $(data);
    // console.log(content)

    $searchLoading.hide()
    
    $.each(data, function(k, v) {
      var model_id = v['model_id'];
      var name = v['name'];
      var image_url = v['image_url'];
      var url = v['url'];
      var location = v['location'][5];
      var html = (
        '<div class="row search-result-row">'+
          '<div class="col-xs-12 col-sm-2 cell search-results-img-div">'
      );
      if (typeof image_url != 'undefined' && image_url != '')
          html += '<img class="search-results-img" src="'+image_url+'" />'
      html += (
          '</div>' +
          '<div class="col-xs-12 col-sm-4 cell">'+
            '<a href="'+url+'" class="restaurant-name-link" '+
              'target="_blank">'+
              name+'</a>'+
          '</div>'+
          '<div class="col-xs-12 col-sm-4 cell">'
      );
      $.each(location, function(i, location_line) {
        html += '<div class="address">'+location_line+'</div>'
      });
      html += (
          '</div>'+
          '<div class="col-xs-12 col-sm-2 cell">'+
            '<input type="number" class="rating search-rating" '+
              'id="search_'+model_id+'" '+
              'name="search-rating" '
      )
      if (model_id in all_ratings)
        html += 'value="'+all_ratings[model_id]['rating']+'" '
      html += (
              'data-clearable="" data-inline '+
              'data-icon-lib="fa" data-active-icon="fa-star" '+
              'data-inactive-icon="fa-star-o" '+
              'data-clearable-icon="fa-trash-o" />'+
          '</div>'+
        '</div>'
      );
      $searchResults.append(html)
      $('#search_'+model_id).data('values', v)
    });
    $('input.search-rating').rating({});
  });
});

function update_user_ratings() {
  $selectedRatings.empty()
  $.each(all_ratings, function(k, v) {
    if (typeof k != 'undefined' && k != '')
      var rating = v['rating']
      var model_id = v['model_id'];
      var name = v['name'];
      var image_url = v['image_url'];
      var url = v['url'];
      var location = v['location'][5];
      var html = (
        '<div class="row rating-row">'+
          '<div class="col-xs-12 col-sm-2 vcenter cell rating-img-div">'
      );
      if (typeof image_url != 'undefined' && image_url != '')
          html += '<img class="rating-img" src="'+image_url+'" />'
      html += (
          '</div>' +
          '<div class="col-xs-12 col-sm-4 vcenter cell">'+
            '<a href="'+url+'" class="restaurant-name-link" target="_blank">'+
              name+'</a>'+
          '</div>'+
          '<div class="col-xs-12 col-sm-4 vcenter cell">'
      );
    $.each(location, function(i, location_line) {
      html += '<div class="address">'+location_line+'</div>'
    });
      html += (
          '</div>'+
          '<div class="col-xs-12 col-sm-2 cell">'+
            '<input type="number" class="rating user-rating" '+
              'id="rating_'+model_id+'" '+
              'name="user-rating" '+
              'value="'+rating+'" '+
              'data-clearable="" data-inline '+
              'data-icon-lib="fa" data-active-icon="fa-star" '+
              'data-inactive-icon="fa-star-o" '+
              'data-clearable-icon="fa-trash-o" />'+
          '</div>'+
        '</div>'
      );
      $selectedRatings.append(html)
      $('#rating_'+model_id).data('values', v)
  });
  $('input.user-rating').rating({});

  $('input.user-rating').change(function() {
    // console.log('input.rating changed')

    var rating = $(this).val();
    var v = $(this).data('values')
    var model_id = v.model_id
    var $search_rating = $('#search_'+model_id)
    if (rating in valid_rating_values) {
      v['rating'] = rating
      all_ratings[model_id] = v
      // update search result ratings if found
      if (typeof $search_rating != 'undefined' && $search_rating != '') {
        $search_rating.val(rating)
        $search_rating.prev().children().first().trigger('mouseout')
      }
    } else if (model_id in all_ratings) {
      delete all_ratings[model_id];
      // update search result ratings if found
      if (typeof $search_rating != 'undefined' && $search_rating != '') {
        $search_rating.prev().children().last().trigger('click')
      }
    }
    // console.log(all_ratings)
    update_user_ratings()
  });


  $selectedRatingsGroup.show();
}

$(document).ajaxComplete(function() {
  // fire when any Ajax requests complete
  $('input.search-rating').change(function() {
    // console.log('input.rating changed')
    $('input.search-rating').each(function() {
      var rating = $(this).val();
      var v = $(this).data('values')
      var model_id = v.model_id
      if (rating in valid_rating_values) {
        v['rating'] = rating
        all_ratings[model_id] = v
      } else if (model_id in all_ratings) {
        delete all_ratings[model_id];
      }
    });
    // console.log(all_ratings)
    update_user_ratings()
  });
});