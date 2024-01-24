let colors = ["red","blue","green","yellow","black"]
console.log(colors.length)

console.log(colors)
colors[2] = "crimson red"



    console.log(colors)


colors.push("pink")
console.log(colors)
colors.pop()
console.log(colors)
colors.unshift("yellow")


var cars = ["mercedes","bmw","honda","crv"]

for(let car of cars){

}
cars.forEach(function(carq){

})

let position = [45.23,98.234];

let[lat,long] = position;

let date  = "10/11/2323"

date_array = date.split("/")

let [dates,month,year] = date_array;
console.log(dates)
console.log(month)

let user = "smith@gmail.com"
user_array = user.split("@")
let [username,] =user_array
console.log(username)

const array = ["one", "two","three","four","five"]
let cap_array = []
array.map((element)=>{
    cap_array.push(element.toUpperCase())
})
console.log(cap_array)