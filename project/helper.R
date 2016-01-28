fix_na <- function(x){
        for(i in 1:ncol(x)){
                if(sum(is.na(x[,i]))>0) x[,i][is.na(x[,i])] <- 0
        }
        x
}