"""Entity Store page — person knowledge management UI.

Redesigned as a native Reflex component matching the settings page style.
Two tabs: People (master-detail side panel) and All Facts (filterable table).
"""

import reflex as rx

from ..state import AppState


# =========================================================================
# MAIN ENTITIES PAGE
# =========================================================================


def entities_page() -> rx.Component:
    """Full entities page with tabbed interface matching settings page."""
    return rx.box(
        rx.flex(
            # Header: back + title + stat badges
            _header(),
            # Status message (action confirmation)
            rx.cond(
                AppState.entity_save_message != "",
                rx.box(
                    rx.text(
                        AppState.entity_save_message,
                        class_name="text-sm",
                    ),
                    class_name="mb-4 px-3 py-2 bg-gray-50 rounded-lg border border-gray-200",
                ),
                rx.fragment(),
            ),
            # Seed status message
            rx.cond(
                AppState.entity_seed_message != "",
                rx.box(
                    rx.text(
                        AppState.entity_seed_message,
                        class_name="text-sm",
                    ),
                    class_name="mb-4 px-3 py-2 bg-gray-50 rounded-lg border border-gray-200",
                ),
                rx.fragment(),
            ),
            # Main tabbed interface
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("👤 People", value="people"),
                    rx.tabs.trigger("📋 All Facts", value="facts"),
                    rx.tabs.trigger("🔀 Suggestions", value="suggestions"),
                    size="2",
                ),
                rx.tabs.content(_people_tab(), value="people", class_name="pt-4"),
                rx.tabs.content(_facts_tab(), value="facts", class_name="pt-4"),
                rx.tabs.content(_suggestions_tab(), value="suggestions", class_name="pt-4"),
                value=AppState.entity_tab,
                on_change=AppState.set_entity_tab,
                default_value="people",
                class_name="mt-2",
            ),
            direction="column",
            class_name="max-w-[1100px] mx-auto w-full px-4 py-6",
        ),
        class_name="h-full overflow-y-auto chat-scroll",
    )


# =========================================================================
# HEADER
# =========================================================================


def _header() -> rx.Component:
    """Header: back button + title + stat badges."""
    return rx.flex(
        # Left: back + title
        rx.flex(
            rx.link(
                rx.icon_button(
                    rx.icon("arrow-left", size=18),
                    variant="ghost",
                    class_name="text-gray-500 hover:text-gray-700",
                ),
                href="/",
            ),
            rx.heading("Entity Store", size="6", class_name="text-gray-800"),
            align="center",
            gap="3",
        ),
        # Right: stat badges
        rx.flex(
            _stat_badge("👤", AppState.entity_stats_persons, "Persons"),
            _stat_badge("📋", AppState.entity_stats_facts, "Facts"),
            _stat_badge("📝", AppState.entity_stats_aliases, "Aliases"),
            _stat_badge("🔗", AppState.entity_stats_relationships, "Rels"),
            gap="3",
            align="center",
        ),
        justify="between",
        align="center",
        class_name="mb-4",
    )


def _stat_badge(icon: str, value: rx.Var, label: str) -> rx.Component:
    """Compact stat badge for the header."""
    return rx.flex(
        rx.text(icon, class_name="text-sm"),
        rx.text(value, class_name="text-sm font-semibold text-gray-700"),
        rx.text(label, class_name="text-xs text-gray-400 hidden sm:inline"),
        align="center",
        gap="1",
        class_name="bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5",
    )


# =========================================================================
# SECTION CARD — shared wrapper (matches settings page)
# =========================================================================


def _section_card(
    title: str,
    icon: str,
    *children: rx.Component,
) -> rx.Component:
    """Wrap content in a visually distinct card with icon + title header."""
    return rx.box(
        rx.box(
            rx.flex(
                rx.flex(
                    rx.icon(icon, size=16, class_name="text-gray-500"),
                    rx.text(title),
                    align="center",
                    gap="2",
                ),
                justify="between",
                align="center",
            ),
            class_name="settings-card-header",
        ),
        *children,
        class_name="settings-card",
    )


# =========================================================================
# TAB: PEOPLE
# =========================================================================


def _people_tab() -> rx.Component:
    """People tab — toolbar + merge bar + conditional master-detail layout."""
    return rx.flex(
        _toolbar(),
        # Merge action bar (shown when merge mode is active)
        rx.cond(
            AppState.entity_merge_mode,
            _merge_action_bar(),
            rx.fragment(),
        ),
        rx.cond(
            AppState.entity_has_detail,
            # Two-column: person list (left) + detail panel (right)
            rx.flex(
                _person_list_column(),
                _person_detail_panel(),
                gap="4",
                class_name="w-full",
            ),
            # Full-width person grid
            _person_grid_full(),
        ),
        direction="column",
        gap="3",
    )


def _merge_action_bar() -> rx.Component:
    """Bar showing merge selection count and merge execute button."""
    return rx.flex(
        rx.flex(
            rx.icon("merge", size=16, class_name="text-purple-500"),
            rx.cond(
                AppState.entity_merge_count > 0,
                rx.flex(
                    rx.text(
                        AppState.entity_merge_count.to(str),  # type: ignore[union-attr]
                        class_name="text-sm font-semibold text-purple-600",
                    ),
                    rx.text("selected", class_name="text-sm text-gray-600"),
                    gap="1",
                    align="center",
                ),
                rx.text(
                    "Click persons to select for merge",
                    class_name="text-sm text-gray-600",
                ),
            ),
            align="center",
            gap="2",
        ),
        rx.text(
            "First selected = merge target (keeps name)",
            class_name="text-xs text-gray-400 italic",
        ),
        rx.button(
            rx.icon("git-merge", size=14, class_name="mr-1"),
            "Merge Selected",
            on_click=AppState.execute_merge,
            disabled=~AppState.entity_can_merge,
            size="2",
            class_name="bg-purple-500 text-white hover:bg-purple-600 disabled:opacity-50",
        ),
        justify="between",
        align="center",
        class_name=(
            "w-full px-4 py-2 bg-purple-50 border border-purple-200 "
            "rounded-lg"
        ),
    )


def _toolbar() -> rx.Component:
    """Search bar + seed + refresh + cleanup + merge buttons."""
    return rx.flex(
        rx.el.input(
            placeholder="Search persons…",
            default_value=AppState.entity_search,
            on_change=AppState.set_entity_search,
            class_name=(
                "flex-1 bg-white border border-gray-200 rounded-lg "
                "px-3 py-2 text-sm text-gray-700 outline-none "
                "placeholder-gray-400"
            ),
        ),
        rx.button(
            rx.icon("search", size=14, class_name="mr-1"),
            "Search",
            on_click=AppState.search_entities,
            variant="outline",
            size="2",
        ),
        rx.button(
            rx.icon("refresh-cw", size=14, class_name="mr-1"),
            "Refresh",
            on_click=AppState.refresh_entities,
            variant="outline",
            size="2",
        ),
        rx.button(
            rx.icon("sprout", size=14, class_name="mr-1"),
            "Seed",
            on_click=AppState.seed_entities,
            loading=AppState.entity_seed_status == "seeding",
            size="2",
            class_name="bg-green-500 text-white hover:bg-green-600",
        ),
        rx.button(
            rx.icon("trash-2", size=14, class_name="mr-1"),
            "Cleanup",
            on_click=AppState.cleanup_entities,
            variant="outline",
            size="2",
            color_scheme="red",
        ),
        # Merge mode toggle
        rx.button(
            rx.icon("merge", size=14, class_name="mr-1"),
            rx.cond(
                AppState.entity_merge_mode,
                "Cancel Merge",
                "Merge",
            ),
            on_click=AppState.toggle_merge_mode,
            variant=rx.cond(AppState.entity_merge_mode, "solid", "outline"),
            size="2",
            color_scheme=rx.cond(AppState.entity_merge_mode, "purple", "gray"),
        ),
        gap="2",
        align="center",
        wrap="wrap",
    )


# =========================================================================
# PERSON GRID (full-width, no selection)
# =========================================================================


def _person_grid_full() -> rx.Component:
    """Full-width person card grid."""
    return rx.cond(
        AppState.entity_loading,
        rx.flex(
            rx.spinner(size="3"),
            rx.text("Loading…", class_name="text-sm text-gray-400 ml-2"),
            align="center",
            justify="center",
            class_name="py-12",
        ),
        rx.cond(
            AppState.entity_persons.length() > 0,  # type: ignore[union-attr]
            rx.box(
                rx.foreach(
                    AppState.entity_persons,
                    _person_card,
                ),
                class_name=(
                    "grid gap-3"
                    " grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
                ),
            ),
            rx.box(
                rx.flex(
                    rx.icon("users", size=40, class_name="text-gray-300"),
                    rx.text(
                        "No persons found",
                        class_name="text-gray-400 mt-2",
                    ),
                    rx.text(
                        "Click 'Seed' to import contacts from WhatsApp",
                        class_name="text-sm text-gray-300 mt-1",
                    ),
                    direction="column",
                    align="center",
                    class_name="py-16",
                ),
            ),
        ),
    )


def _person_card(person: dict) -> rx.Component:
    """Single person card for the grid — with merge checkbox when in merge mode."""
    is_selected_for_merge = AppState.entity_merge_selection.contains(person["id"])
    merge_index = _merge_index_label(person["id"])

    return rx.box(
        # Merge mode: checkbox overlay
        rx.cond(
            AppState.entity_merge_mode,
            rx.flex(
                rx.cond(
                    is_selected_for_merge,
                    rx.flex(
                        rx.icon("check", size=14, class_name="text-white"),
                        class_name=(
                            "w-6 h-6 rounded-full bg-purple-500 "
                            "flex items-center justify-center shrink-0"
                        ),
                    ),
                    rx.box(
                        class_name=(
                            "w-6 h-6 rounded-full border-2 border-gray-300 "
                            "shrink-0"
                        ),
                    ),
                ),
                rx.cond(
                    is_selected_for_merge,
                    rx.text(
                        merge_index,
                        class_name="text-xs text-purple-500 font-bold",
                    ),
                    rx.fragment(),
                ),
                align="center",
                gap="1",
                class_name="absolute top-2 right-2",
            ),
            rx.fragment(),
        ),
        rx.text(
            person["canonical_name"],
            class_name="text-sm font-semibold text-gray-800 truncate",
        ),
        # Aliases preview (clean comma-separated names)
        rx.cond(
            person["aliases_preview"] != "",
            rx.text(
                person["aliases_preview"],
                class_name="text-xs text-gray-400 truncate mt-1",
            ),
            rx.fragment(),
        ),
        # Stats row
        rx.flex(
            rx.flex(
                rx.icon("file-text", size=12, class_name="text-gray-400"),
                rx.text(
                    person["fact_count"],
                    class_name="text-xs text-gray-500",
                ),
                rx.text("facts", class_name="text-xs text-gray-400"),
                align="center",
                gap="1",
            ),
            rx.flex(
                rx.icon("tag", size=12, class_name="text-gray-400"),
                rx.text(
                    person["alias_count"],
                    class_name="text-xs text-gray-500",
                ),
                rx.text("aliases", class_name="text-xs text-gray-400"),
                align="center",
                gap="1",
            ),
            gap="3",
            class_name="mt-2",
        ),
        on_click=rx.cond(
            AppState.entity_merge_mode,
            AppState.toggle_merge_selection(person["id"]),
            AppState.select_entity(person["id"]),
        ),
        class_name=rx.cond(
            is_selected_for_merge,
            (
                "settings-card cursor-pointer border-purple-400 bg-purple-50 "
                "transition-all duration-150 hover:-translate-y-0.5 relative"
            ),
            (
                "settings-card cursor-pointer hover:border-accent "
                "transition-all duration-150 hover:-translate-y-0.5 relative"
            ),
        ),
    )


def _merge_index_label(person_id: rx.Var) -> rx.Var:
    """Compute a label like '① target' or '②' for merge selection order.

    Uses a simple conditional since rx.foreach doesn't support index tracking.
    """
    return rx.cond(
        AppState.entity_merge_selection.length() > 0,  # type: ignore[union-attr]
        rx.cond(
            AppState.entity_merge_selection[0] == person_id,
            "① target",
            "②+",
        ),
        "",
    )


# =========================================================================
# PERSON LIST COLUMN (narrow, when detail is open)
# =========================================================================


def _person_list_column() -> rx.Component:
    """Narrow scrollable person list for master-detail layout."""
    return rx.box(
        rx.foreach(
            AppState.entity_persons,
            _person_list_item,
        ),
        class_name=(
            "w-[280px] min-w-[280px] max-h-[calc(100vh-280px)] "
            "overflow-y-auto space-y-2 pr-2"
        ),
    )


def _person_list_item(person: dict) -> rx.Component:
    """Compact person item for the side list — with merge checkbox."""
    is_selected = AppState.entity_selected_id == person["id"].to(int)
    is_merge_selected = AppState.entity_merge_selection.contains(person["id"])

    return rx.box(
        rx.flex(
            # Merge checkbox (when in merge mode)
            rx.cond(
                AppState.entity_merge_mode,
                rx.cond(
                    is_merge_selected,
                    rx.flex(
                        rx.icon("check", size=10, class_name="text-white"),
                        class_name=(
                            "w-5 h-5 rounded-full bg-purple-500 "
                            "flex items-center justify-center shrink-0"
                        ),
                    ),
                    rx.box(
                        class_name=(
                            "w-5 h-5 rounded-full border-2 border-gray-300 "
                            "shrink-0"
                        ),
                    ),
                ),
                rx.fragment(),
            ),
            rx.box(
                rx.text(
                    person["canonical_name"],
                    class_name="text-sm font-medium text-gray-800 truncate",
                ),
                rx.flex(
                    rx.flex(
                        rx.icon("file-text", size=11, class_name="text-gray-400"),
                        rx.text(
                            person["fact_count"],
                            class_name="text-xs text-gray-400",
                        ),
                        align="center",
                        gap="1",
                    ),
                    rx.flex(
                        rx.icon("tag", size=11, class_name="text-gray-400"),
                        rx.text(
                            person["alias_count"],
                            class_name="text-xs text-gray-400",
                        ),
                        align="center",
                        gap="1",
                    ),
                    gap="3",
                    class_name="mt-0.5",
                ),
                class_name="flex-1 min-w-0",
            ),
            align="center",
            gap="2",
        ),
        on_click=rx.cond(
            AppState.entity_merge_mode,
            AppState.toggle_merge_selection(person["id"]),
            AppState.select_entity(person["id"]),
        ),
        class_name=rx.cond(
            is_merge_selected,
            (
                "px-3 py-2 rounded-lg border border-purple-400 bg-purple-50 "
                "cursor-pointer"
            ),
            rx.cond(
                is_selected,
                (
                    "px-3 py-2 rounded-lg border border-accent bg-green-50 "
                    "cursor-pointer"
                ),
                (
                    "px-3 py-2 rounded-lg border border-gray-200 bg-white "
                    "cursor-pointer hover:border-gray-300 hover:bg-gray-50 "
                    "transition-colors duration-150"
                ),
            ),
        ),
    )


# =========================================================================
# PERSON DETAIL PANEL
# =========================================================================


def _person_detail_panel() -> rx.Component:
    """Detail side panel showing full person info."""
    return rx.box(
        rx.cond(
            AppState.entity_detail_loading,
            rx.flex(
                rx.spinner(size="3"),
                rx.text("Loading…", class_name="text-sm text-gray-400 ml-2"),
                align="center",
                justify="center",
                class_name="py-12",
            ),
            rx.flex(
                # Header with name + actions
                _detail_header(),
                # Contact info
                _detail_contact_info(),
                # Grouped facts
                _detail_facts(),
                # Aliases
                _detail_aliases(),
                # Relationships
                _detail_relationships(),
                # Add new fact form
                _detail_add_fact(),
                direction="column",
                gap="3",
            ),
        ),
        class_name=(
            "flex-1 max-h-[calc(100vh-280px)] overflow-y-auto "
            "border-l border-gray-200 pl-4"
        ),
    )


def _detail_header() -> rx.Component:
    """Person name, bilingual name button, close button, delete button."""
    return rx.flex(
        rx.heading(
            AppState.entity_detail_name,
            size="5",
            class_name="text-gray-800",
        ),
        rx.flex(
            # Bilingual name merge button
            rx.tooltip(
                rx.icon_button(
                    rx.icon("languages", size=14),
                    on_click=AppState.update_entity_display_name,
                    variant="outline",
                    size="1",
                    class_name="text-blue-500 hover:text-blue-700",
                ),
                content="Merge Hebrew + English names",
            ),
            rx.button(
                rx.icon("trash-2", size=14, class_name="mr-1"),
                "Delete",
                on_click=AppState.delete_entity,
                variant="outline",
                size="1",
                color_scheme="red",
            ),
            rx.icon_button(
                rx.icon("x", size=18),
                on_click=AppState.close_entity_detail,
                variant="ghost",
                class_name="text-gray-400 hover:text-gray-600",
            ),
            gap="2",
            align="center",
        ),
        justify="between",
        align="center",
        class_name="mb-2",
    )


def _detail_contact_info() -> rx.Component:
    """WhatsApp ID and phone display."""
    return rx.cond(
        (AppState.entity_detail_phone != "") | (AppState.entity_detail_whatsapp != ""),
        rx.flex(
            rx.cond(
                AppState.entity_detail_phone != "",
                rx.flex(
                    rx.icon("phone", size=14, class_name="text-gray-400"),
                    rx.text(
                        AppState.entity_detail_phone,
                        class_name="text-sm text-gray-600",
                    ),
                    align="center",
                    gap="1.5",
                ),
                rx.fragment(),
            ),
            rx.cond(
                AppState.entity_detail_whatsapp != "",
                rx.flex(
                    rx.icon("message-circle", size=14, class_name="text-green-500"),
                    rx.text(
                        AppState.entity_detail_whatsapp,
                        class_name="text-sm text-gray-600",
                    ),
                    align="center",
                    gap="1.5",
                ),
                rx.fragment(),
            ),
            gap="4",
            class_name="mb-2 text-sm",
        ),
        rx.fragment(),
    )


# =========================================================================
# DETAIL: GROUPED FACTS
# =========================================================================


def _detail_facts() -> rx.Component:
    """Render grouped facts with category headers and inline edit/delete."""
    return rx.cond(
        AppState.entity_facts_grouped.length() > 0,  # type: ignore[union-attr]
        rx.box(
            rx.foreach(
                AppState.entity_facts_grouped,
                _render_fact_item,
            ),
        ),
        rx.box(
            rx.text(
                "No facts recorded yet",
                class_name="text-sm text-gray-400 italic py-4",
            ),
        ),
    )


def _render_fact_item(item: dict) -> rx.Component:
    """Render either a category header or a fact row."""
    return rx.cond(
        item["type"] == "header",
        _fact_category_header(item),
        _fact_row(item),
    )


def _fact_category_header(item: dict) -> rx.Component:
    """Category header for a fact group."""
    return rx.box(
        rx.flex(
            rx.icon(item["icon"], size=16, class_name="text-gray-500"),
            rx.text(
                item["category"],
                class_name="text-sm font-semibold text-gray-600",
            ),
            align="center",
            gap="2",
        ),
        class_name="settings-card-header mt-2",
    )


def _fact_row(item: dict) -> rx.Component:
    """Single fact row with label, value, confidence, source, edit/delete."""
    is_editing = AppState.entity_editing_fact_key == item["fact_key"]

    return rx.cond(
        is_editing,
        _fact_row_edit(item),
        _fact_row_display(item),
    )


def _fact_row_display(item: dict) -> rx.Component:
    """Fact row in display mode."""
    return rx.flex(
        # Label
        rx.text(
            item["label"],
            class_name="text-sm text-gray-500 w-[120px] shrink-0",
        ),
        # Value
        rx.text(
            item["value"],
            class_name="text-sm text-gray-800 flex-1",
        ),
        # Confidence badge
        rx.cond(
            item["confidence"] != "",
            rx.text(
                item["confidence"],
                class_name=rx.cond(
                    item["confidence"].contains("100") | item["confidence"].contains("9") | item["confidence"].contains("8") | item["confidence"].contains("7"),  # type: ignore[union-attr]
                    "text-xs text-green-600 bg-green-50 px-1.5 py-0.5 rounded",
                    rx.cond(
                        item["confidence"].contains("6") | item["confidence"].contains("5") | item["confidence"].contains("4"),  # type: ignore[union-attr]
                        "text-xs text-yellow-600 bg-yellow-50 px-1.5 py-0.5 rounded",
                        "text-xs text-red-600 bg-red-50 px-1.5 py-0.5 rounded",
                    ),
                ),
            ),
            rx.fragment(),
        ),
        # Source icon
        rx.cond(
            item["source_type"] != "",
            _source_icon(item["source_type"]),
            rx.fragment(),
        ),
        # Edit button
        rx.icon_button(
            rx.icon("pencil", size=12),
            on_click=AppState.start_edit_fact(item["fact_key"], item["value"]),
            variant="ghost",
            size="1",
            class_name="text-gray-300 hover:text-gray-500",
        ),
        # Delete button
        rx.icon_button(
            rx.icon("trash-2", size=12),
            on_click=AppState.delete_entity_fact(item["fact_key"]),
            variant="ghost",
            size="1",
            class_name="text-gray-300 hover:text-red-500",
        ),
        align="center",
        gap="2",
        class_name="px-4 py-2 border-b border-gray-100 last:border-b-0 hover:bg-gray-50",
    )


def _fact_row_edit(item: dict) -> rx.Component:
    """Fact row in edit mode."""
    return rx.flex(
        # Label
        rx.text(
            item["label"],
            class_name="text-sm text-gray-500 w-[120px] shrink-0",
        ),
        # Editable input
        rx.el.input(
            type="text",
            default_value=AppState.entity_editing_fact_value,
            on_change=AppState.set_entity_editing_fact_value,
            auto_focus=True,
            class_name=(
                "flex-1 bg-white border border-accent rounded-lg "
                "px-2 py-1 text-sm text-gray-700 outline-none"
            ),
        ),
        # Save
        rx.icon_button(
            rx.icon("check", size=14),
            on_click=AppState.save_entity_fact_edit,
            variant="solid",
            size="1",
            color_scheme="green",
        ),
        # Cancel
        rx.icon_button(
            rx.icon("x", size=14),
            on_click=AppState.cancel_edit_fact,
            variant="ghost",
            size="1",
            class_name="text-gray-400 hover:text-gray-600",
        ),
        align="center",
        gap="2",
        class_name="px-4 py-2 border-b border-gray-100 bg-green-50",
    )


def _source_icon(source_type: rx.Var) -> rx.Component:
    """Small icon indicating the source of a fact."""
    return rx.tooltip(
        rx.icon(
            rx.cond(
                source_type == "whatsapp",
                "message-circle",
                rx.cond(
                    source_type == "paperless",
                    "file-text",
                    rx.cond(
                        source_type == "manual",
                        "pencil",
                        "zap",
                    ),
                ),
            ),
            size=12,
            class_name=rx.cond(
                source_type == "whatsapp",
                "text-green-400",
                rx.cond(
                    source_type == "paperless",
                    "text-blue-400",
                    rx.cond(
                        source_type == "manual",
                        "text-purple-400",
                        "text-gray-400",
                    ),
                ),
            ),
        ),
        content=source_type,
    )


# =========================================================================
# DETAIL: ALIASES
# =========================================================================


def _detail_aliases() -> rx.Component:
    """Aliases section with removable bubbles + add form."""
    return _section_card(
        "Aliases", "tag",
        # Alias bubbles
        rx.cond(
            AppState.entity_aliases_list.length() > 0,  # type: ignore[union-attr]
            rx.flex(
                rx.foreach(
                    AppState.entity_aliases_list,
                    _alias_bubble,
                ),
                wrap="wrap",
                gap="2",
                class_name="mb-3",
            ),
            rx.text(
                "No aliases",
                class_name="text-sm text-gray-400 italic mb-3",
            ),
        ),
        # Add alias form
        rx.flex(
            rx.el.input(
                type="text",
                placeholder="Add alias…",
                value=AppState.entity_new_alias,
                on_change=AppState.set_entity_new_alias,
                class_name=(
                    "flex-1 bg-white border border-gray-200 rounded-lg "
                    "px-3 py-1.5 text-sm text-gray-700 outline-none "
                    "focus:border-accent"
                ),
            ),
            rx.button(
                rx.icon("plus", size=14, class_name="mr-1"),
                "Add",
                on_click=AppState.add_entity_alias,
                size="1",
                class_name="bg-accent text-white hover:bg-accent-hover shrink-0",
            ),
            align="center",
            gap="2",
        ),
    )


def _alias_bubble(alias: dict) -> rx.Component:
    """Removable alias bubble."""
    return rx.box(
        rx.flex(
            rx.icon(
                "x",
                size=12,
                class_name="shrink-0 cursor-pointer opacity-50 hover:opacity-100",
                on_click=AppState.delete_entity_alias(alias["id"]),
            ),
            rx.text(
                alias["alias"],
                class_name="text-sm",
            ),
            rx.cond(
                alias["script"] != "",
                rx.text(
                    alias["script"],
                    class_name="text-[10px] text-gray-400",
                ),
                rx.fragment(),
            ),
            align="center",
            gap="1.5",
        ),
        class_name=(
            "px-2.5 py-1 rounded-full border border-gray-200 "
            "bg-gray-50 inline-flex text-gray-700"
        ),
    )


# =========================================================================
# DETAIL: RELATIONSHIPS
# =========================================================================


def _detail_relationships() -> rx.Component:
    """Relationships section."""
    return rx.cond(
        AppState.entity_relationships_list.length() > 0,  # type: ignore[union-attr]
        _section_card(
            "Relationships", "link",
            rx.flex(
                rx.foreach(
                    AppState.entity_relationships_list,
                    _relationship_badge,
                ),
                wrap="wrap",
                gap="2",
            ),
        ),
        rx.fragment(),
    )


def _relationship_badge(rel: dict) -> rx.Component:
    """Single relationship display badge."""
    return rx.flex(
        rx.text(
            rel["type"],
            class_name="text-xs text-gray-500 uppercase tracking-wider",
        ),
        rx.icon("arrow-right", size=12, class_name="text-gray-400"),
        rx.text(
            rel["related_name"],
            class_name="text-sm font-medium text-gray-700",
        ),
        align="center",
        gap="1.5",
        class_name=(
            "px-3 py-1.5 rounded-lg border border-gray-200 bg-gray-50"
        ),
    )


# =========================================================================
# DETAIL: ADD FACT FORM
# =========================================================================


def _detail_add_fact() -> rx.Component:
    """Add new fact form at the bottom of the detail panel."""
    return _section_card(
        "Add Fact", "circle-plus",
        rx.flex(
            rx.el.input(
                type="text",
                placeholder="Key (e.g. birth_date)",
                value=AppState.entity_new_fact_key,
                on_change=AppState.set_entity_new_fact_key,
                class_name=(
                    "bg-white border border-gray-200 rounded-lg "
                    "px-3 py-2 text-sm text-gray-700 outline-none "
                    "focus:border-accent w-[160px]"
                ),
            ),
            rx.el.input(
                type="text",
                placeholder="Value",
                value=AppState.entity_new_fact_value,
                on_change=AppState.set_entity_new_fact_value,
                class_name=(
                    "flex-1 bg-white border border-gray-200 rounded-lg "
                    "px-3 py-2 text-sm text-gray-700 outline-none "
                    "focus:border-accent"
                ),
            ),
            rx.button(
                rx.icon("plus", size=14, class_name="mr-1"),
                "Add Fact",
                on_click=AppState.add_entity_fact,
                size="2",
                class_name="bg-accent text-white hover:bg-accent-hover shrink-0",
            ),
            align="center",
            gap="2",
        ),
    )


# =========================================================================
# TAB: ALL FACTS
# =========================================================================


def _facts_tab() -> rx.Component:
    """All Facts tab — global fact table with key filter."""
    return rx.flex(
        # Controls
        rx.flex(
            rx.el.select(
                rx.el.option("All fact keys", value=""),
                rx.foreach(
                    AppState.entity_fact_keys,
                    lambda k: rx.el.option(k, value=k),
                ),
                value=AppState.entity_fact_key_filter,
                on_change=AppState.set_entity_fact_key_filter,
                class_name=(
                    "min-w-[180px] bg-white border border-gray-200 rounded-lg "
                    "px-3 py-2 text-sm text-gray-700 outline-none"
                ),
            ),
            rx.button(
                rx.icon("search", size=14, class_name="mr-1"),
                "Load Facts",
                on_click=AppState.load_all_entity_facts,
                size="2",
                variant="outline",
            ),
            gap="2",
            align="center",
            class_name="mb-3",
        ),
        # Facts table
        rx.cond(
            AppState.entity_all_facts.length() > 0,  # type: ignore[union-attr]
            rx.box(
                rx.el.table(
                    rx.el.thead(
                        rx.el.tr(
                            rx.el.th("Person", class_name="px-3 py-2 text-left text-xs text-gray-500 font-medium"),
                            rx.el.th("Key", class_name="px-3 py-2 text-left text-xs text-gray-500 font-medium"),
                            rx.el.th("Value", class_name="px-3 py-2 text-left text-xs text-gray-500 font-medium"),
                            rx.el.th("Confidence", class_name="px-3 py-2 text-left text-xs text-gray-500 font-medium"),
                            rx.el.th("Source", class_name="px-3 py-2 text-left text-xs text-gray-500 font-medium"),
                            class_name="bg-gray-50 border-b border-gray-200",
                        ),
                    ),
                    rx.el.tbody(
                        rx.foreach(
                            AppState.entity_all_facts,
                            _all_facts_row,
                        ),
                    ),
                    class_name="w-full text-sm",
                ),
                class_name="border border-gray-200 rounded-lg overflow-hidden",
            ),
            rx.text(
                "Click 'Load Facts' to view all facts across all persons.",
                class_name="text-sm text-gray-400 italic py-8 text-center",
            ),
        ),
        direction="column",
    )


def _all_facts_row(fact: dict) -> rx.Component:
    """Single row in the All Facts table."""
    return rx.el.tr(
        rx.el.td(
            rx.text(
                fact["person_name"],
                class_name="font-medium text-gray-800",
            ),
            class_name="px-3 py-2",
        ),
        rx.el.td(
            rx.text(
                fact["fact_key"],
                class_name="text-accent font-mono text-xs",
            ),
            class_name="px-3 py-2",
        ),
        rx.el.td(
            rx.text(fact["fact_value"]),
            class_name="px-3 py-2",
        ),
        rx.el.td(
            rx.cond(
                fact["confidence"] != "",
                rx.text(
                    fact["confidence"],
                    class_name="text-xs text-gray-500",
                ),
                rx.text("—", class_name="text-gray-300"),
            ),
            class_name="px-3 py-2",
        ),
        rx.el.td(
            rx.cond(
                fact["source_type"] != "",
                _source_icon(fact["source_type"]),
                rx.fragment(),
            ),
            class_name="px-3 py-2",
        ),
        on_click=AppState.select_entity(fact["person_id"]),
        class_name=(
            "border-b border-gray-100 cursor-pointer hover:bg-gray-50 "
            "transition-colors duration-100"
        ),
    )


# =========================================================================
# TAB: MERGE SUGGESTIONS
# =========================================================================


def _suggestions_tab() -> rx.Component:
    """Merge suggestions tab — shows potential duplicates with one-click merge."""
    return rx.flex(
        # Controls
        rx.flex(
            rx.button(
                rx.icon("scan-search", size=14, class_name="mr-1"),
                "Find Duplicates",
                on_click=AppState.load_merge_candidates,
                loading=AppState.entity_candidates_loading,
                size="2",
                class_name="bg-purple-500 text-white hover:bg-purple-600",
            ),
            rx.text(
                "Scans for persons sharing phone, email, WhatsApp ID, aliases, or names",
                class_name="text-xs text-gray-400 italic ml-2",
            ),
            gap="2",
            align="center",
            class_name="mb-4",
        ),
        # Candidate list
        rx.cond(
            AppState.entity_merge_candidates.length() > 0,  # type: ignore[union-attr]
            rx.box(
                rx.foreach(
                    AppState.entity_merge_candidates,
                    _suggestion_card,
                ),
                class_name="space-y-3",
            ),
            rx.cond(
                AppState.entity_candidates_loading,
                rx.flex(
                    rx.spinner(size="3"),
                    rx.text("Scanning…", class_name="text-sm text-gray-400 ml-2"),
                    align="center",
                    justify="center",
                    class_name="py-12",
                ),
                rx.box(
                    rx.flex(
                        rx.icon("circle-check", size=40, class_name="text-green-300"),
                        rx.text(
                            "No merge suggestions yet",
                            class_name="text-gray-400 mt-2",
                        ),
                        rx.text(
                            "Click 'Find Duplicates' to scan for potential merges",
                            class_name="text-sm text-gray-300 mt-1",
                        ),
                        direction="column",
                        align="center",
                        class_name="py-16",
                    ),
                ),
            ),
        ),
        direction="column",
    )


def _suggestion_card(candidate: dict) -> rx.Component:
    """Single merge suggestion card with reason, persons, and merge button."""
    return rx.box(
        rx.flex(
            # Left: reason + person details
            rx.box(
                rx.flex(
                    rx.text(
                        candidate["reason"],
                        class_name="text-sm font-semibold text-purple-700",
                    ),
                    rx.flex(
                        rx.text(
                            candidate["count"],
                            class_name="text-xs text-gray-400",
                        ),
                        rx.text(
                            "persons",
                            class_name="text-xs text-gray-400",
                        ),
                        gap="1",
                        class_name="ml-2",
                    ),
                    align="center",
                    gap="1",
                ),
                rx.text(
                    candidate["names"],
                    class_name="text-sm text-gray-800 mt-1 font-medium",
                ),
                rx.text(
                    candidate["details"],
                    class_name="text-xs text-gray-500 mt-0.5 truncate",
                ),
                class_name="flex-1 min-w-0",
            ),
            # Right: merge button
            rx.button(
                rx.icon("git-merge", size=14, class_name="mr-1"),
                "Merge",
                on_click=AppState.merge_candidate_group(
                    candidate["target_id"],
                    candidate["source_ids"],
                ),
                size="2",
                class_name=(
                    "bg-purple-500 text-white hover:bg-purple-600 shrink-0"
                ),
            ),
            justify="between",
            align="center",
            gap="3",
        ),
        class_name=(
            "px-4 py-3 border border-purple-200 bg-purple-50 "
            "rounded-lg hover:border-purple-300 transition-colors"
        ),
    )
