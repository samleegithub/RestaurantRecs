db.getCollection('reviews').find({page: {$exists: true}}, {page: 1}).forEach(function (x) {
   db.getCollection('reviews').update({ _id: x._id },
      {$set: {
        page: NumberInt(x.page)
      }});
});