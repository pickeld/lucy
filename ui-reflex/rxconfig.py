import reflex as rx

config = rx.Config(
    app_name="ui_reflex",
    plugins=[
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.SitemapPlugin(),
    ],
)
