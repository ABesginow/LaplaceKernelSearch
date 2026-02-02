import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    x = 42

    return (x,)


@app.cell
def _():
    return


@app.cell
def definitions(x):
    if x == 42:
        y = 42
    else:
        y = 17

    return (y,)


@app.cell
def _(y):
    # Definitions 2
    y
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
