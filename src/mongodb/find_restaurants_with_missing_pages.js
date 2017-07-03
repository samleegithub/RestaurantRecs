db.getCollection('reviews').aggregate(
        [
            {'$group': {
                '_id': {'yelp_id': '$yelp_id', 'num_pages': '$num_pages'},
                //'first_page_saved': {'$min': '$page'},
                //'last_page_saved': {'$max': '$page'},
                'count': {'$sum': 1}
            }},
            {'$project' : {
                '_id' : 1,
                'last_page_saved' : 1,
                'c_cmp' : {'$cmp' : ['$count', '$_id.num_pages']}
            }},
            {'$match' : {'c_cmp' : -1}}
        ]
    )