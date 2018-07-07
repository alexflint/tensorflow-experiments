
def main():
    data = np.arange(24).astype(np.float32).reshape((-1, 2, 2))
    slices = tf.data.Dataset.from_tensor_slices(data)
    it = slices.make_one_shot_iterator().get_next()

    model = tf.layers.Dense(units=1)
    out = model(it)

    training_features = {
        "bedrooms": [[3.], [4.], [2.], [1.]],
        "bathrooms": [[2.], [3.5], [1.], [2.]],
        "city": ["San Francisco", "San Jose", "Daly City", "San Francisco"],
    }

    bedrooms_col = tf.feature_column.numeric_column("bedrooms")
    bathrooms_col = tf.feature_column.numeric_column("bathrooms")
    city_col = tf.feature_column.categorical_column_with_vocabulary_list(
        "city", CITIES)
    city_col = tf.feature_column.indicator_column(city_col)

    cols = [bedrooms_col, bathrooms_col, city_col]
    inputs = tf.feature_column.input_layer(training_features, cols)

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()

    s = tf.Session()
    s.run((var_init, table_init))
    first = s.run(inputs)
    print(first)

if __name__ == "__main__":
    main()
