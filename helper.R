missClass = function(values,prediction){
        sum(((prediction > 0.5)*1) != values)/length(values)
}