const tf = require("@tensorflow/tfjs-node");
// const tf = require("@tensorflow/tfjs");
const movies = require("./data/movielens100k_details.json");

let isLoadingModel = false;
let modelLoaded = false;
let model;

function loadData() {
  const movie_arr = [];
  for (let i = 0; i < movies.length; i++) {
    movie_arr.push([movies[i]["movie_id"]]);
  }
  return movie_arr;
}

async function loadModel() {
  console.log("Loading Model...");
  // const model = await tf.loadLayersModel(
  //   "https://raw.githubusercontent.com/hafezdeldaffa/week3-ml-movielens/master/model/model.json",
  //   false
  // );
  const model = await tf.loadLayersModel(
    `file://${__dirname}/model/model.json`,
    false
  );
  console.log("Model Loaded Successfull");
  modelLoaded = true;
  return model;
  // model.summary()
}

const movie_arr = tf.tensor(loadData());
const movie_len = movies.length;

exports.recommend = async function recommend(userId) {
  console.log(modelLoaded);
  let user = tf.fill([movie_len], Number(userId));
  let movie_in_js_array = movie_arr.arraySync();
  let recommendations = [];
  if (!modelLoaded) {
    if (!isLoadingModel) {
      isLoadingModel = true;
      model = await loadModel();
    }
    for (let i = 0; i < 6; i++) {
      recommendations.push(movies[i]); //Push movie with highest prediction probability
    }
  } else {
    console.log(`Recommending for User: ${userId}`);
    pred_tensor = await model.predict([movie_arr, user]).reshape([movie_len]);
    pred = pred_tensor.arraySync();

    for (let i = 0; i < 6; i++) {
      max = pred_tensor.argMax().arraySync();
      recommendations.push(movies[max]); //Push movie with highest prediction probability
      pred.splice(max, 1); //drop from array
      pred_tensor = tf.tensor(pred); //create a new tensor
    }
  }
  console.log("returned");
  return [recommendations, modelLoaded];
};
