db.getCollection('reviews').update(
    // query 
    {
    },
    
    // update 
    {
        '$inc' : {'page' : 1}
    },
    
    // options 
    {
        "multi" : true,  // update only one document 
        "upsert" : false  // insert a new document, if no existing document match the query 
    }
);