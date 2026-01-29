# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import logging
import sys
import typing
import warnings

import click
from dotenv import load_dotenv
import uvicorn

from google.adk.cli import fast_api
from google.adk.cli.utils import logs


warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\].*",
    category=UserWarning
)
os.environ["ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS"] = "True"
load_dotenv(os.getenv("ENV_FILE", ".env"))

sys.path.insert(0, os.path.dirname(__file__))

LOG_LEVELS = click.Choice(
    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    case_sensitive=False,
)

@click.command()
@click.argument(
    "agents_dir",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=os.getcwd(),
)
@click.option(
    "--host",
    type=str,
    help="Optional. The binding host of the server",
    default="127.0.0.1",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    help="Optional. The port of the server",
    default=os.getenv("PORT", 8000),
    show_default=True
)
@click.option(
    "--allow_origins",
    help="Optional. Any additional origins to allow for CORS.",
    multiple=True,
)
@click.option(
    "--eval_storage_uri",
    type=str,
    help=(
        "Optional. The evals storage URI to store agent evals,"
        " supported URIs: gs://<bucket name>."
    ),
    default=None,
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    show_default=True,
    default=False,
    help="Enable verbose (DEBUG) logging. Shortcut for --log_level DEBUG.",
)
@click.option(
    "--log_level",
    type=LOG_LEVELS,
    default="INFO",
    help="Optional. Set the logging level",
)
@click.option(
    "--trace_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable cloud trace for telemetry.",
)
@click.option(
    "--otel_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Whether to write OTel data to Google Cloud"
        " Observability services - Cloud Trace and Cloud Logging."
    ),
)
@click.option(
    "--session_service_uri",
    help=(
        """Optional. The URI of the session service.
      - Use 'agentengine://<agent_engine>' to connect to Agent Engine
        sessions. <agent_engine> can either be the full qualified resource
        name 'projects/abc/locations/us-central1/reasoningEngines/123' or
        the resource id '123'.
      - Use 'sqlite://<path_to_sqlite_file>' to connect to an aio-sqlite
        based session service, which is good for local development.
      - Use 'postgresql://<user>:<password>@<host>:<port>/<database_name>'
        to connect to a PostgreSQL DB.
      - See https://docs.sqlalchemy.org/en/20/core/engines.html#backend-specific-urls
        for more details on other database URIs supported by SQLAlchemy."""
    ),
)
@click.option(
    "--artifact_service_uri",
    type=str,
    help=(
        "Optional. The URI of the artifact service,"
        " supported URIs: gs://<bucket name> for GCS artifact service."
    ),
    default=None,
)
@click.option(
    "--memory_service_uri",
    type=str,
    help=("""Optional. The URI of the memory service."""),
    default=None,
)
@click.option(
    "--with_web_ui",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable ADK Web UI.",
)
@click.option(
    "--url_prefix",
    type=str,
    default=None,
    help="Optional. The URL prefix for the ADK API server.",
)
@click.option(
    "--extra_plugins",
    multiple=True,
    default=None,
    help="Optional. Extra plugins to load.",
)
@click.option(
    "--a2a",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable A2A endpoint.",
)
@click.option(
    "--publish_agent_info",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to publish agent info via a /apps/{agent_name}/agent-info endpoint.",
)
def main(
    agents_dir: str,
    host: str,
    port: int,
    allow_origins: typing.Optional[typing.List[str]],
    eval_storage_uri: typing.Optional[str] = None,
    verbose: bool = False,
    log_level: str = "INFO",
    trace_to_cloud: bool = False,
    otel_to_cloud: bool = False,
    session_service_uri: typing.Optional[str] = None,
    artifact_service_uri: typing.Optional[str] = None,
    memory_service_uri: typing.Optional[str] = None,
    with_web_ui: typing.Optional[bool] = None,
    url_prefix: typing.Optional[str] = None,
    extra_plugins: typing.Optional[typing.List[str]] = None,
    a2a: bool = False,
    publish_agent_info: bool = False,
):
    """Starts a FastAPI server for agents.

    AGENTS_DIR: The directory of agents, where each sub-directory is a single
    agent.
    """
    if verbose:
        log_level = "DEBUG"

    logs.setup_adk_logger(getattr(logging, log_level.upper()))

    reload = False
    reload_agents = False

    folders_to_delete = []
    files_to_delete = []
    agent_loader = None

    if a2a:
        from pathlib import Path
        from a2a.types import AgentCapabilities
        from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
        from google.adk.cli.utils.agent_loader import AgentLoader
        from google.adk.apps import App

        if not agent_loader:
            agent_loader = AgentLoader(agents_dir)
        agents = agent_loader.list_agents()
        if len(agents) == 0:
            agents = ["agent"]
        for agent_name in agents:
            agent_card_dir = Path(agents_dir) / agent_name
            if not agent_card_dir.exists():
                agent_card_dir.mkdir(exist_ok=True)
                folders_to_delete.append(agent_card_dir)
            card_file = agent_card_dir / "agent.json"
            if card_file.exists():
                continue
            files_to_delete.append(card_file)
            agent = agent_loader.load_agent(agent_name)
            if isinstance(agent, App):
                agent = agent.root_agent
            card_builder = AgentCardBuilder(
                agent=agent,
                rpc_url=f"http://127.0.0.1/a2a/{agent_name}",
                capabilities=AgentCapabilities(streaming=True)
            )
            agent_card = asyncio.run(card_builder.build())
            card_json = agent_card.model_dump_json(indent=2)
            card_file.write_text(card_json)

    app = fast_api.get_fast_api_app(
        agents_dir=agents_dir,
        session_service_uri=session_service_uri,
        artifact_service_uri=artifact_service_uri,
        memory_service_uri=memory_service_uri,
        eval_storage_uri=eval_storage_uri,
        allow_origins=allow_origins,
        web=with_web_ui or False,
        trace_to_cloud=trace_to_cloud,
        otel_to_cloud=otel_to_cloud,
        a2a=a2a,
        host=host,
        port=port,
        url_prefix=url_prefix,
        reload_agents=reload_agents,
        extra_plugins=extra_plugins,
    )

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    if publish_agent_info:
        from google.adk.apps import App
        from google.adk.cli.utils.agent_loader import AgentLoader
        from google.adk.tools import AgentTool
        from google.adk.tools.base_toolset import BaseToolset
        from google.adk.tools.function_tool import FunctionTool
        from google.genai import types as genai_types
        if not agent_loader:
            agent_loader = AgentLoader(agents_dir=agents_dir)

        @app.get("/apps/{agent_name}/agent-info")
        async def agent_info(agent_name: str) -> typing.Dict[str, typing.Any]:
            async def get_tool_functions(tools):
                funcs = []
                for tool in tools:
                    if isinstance(tool, typing.Callable) or isinstance(tool, FunctionTool):
                        funcs.append(genai_types.Tool(
                            function_declarations=[
                                genai_types.FunctionDeclaration.from_callable_with_api_option(
                                    callable=tool if isinstance(tool, typing.Callable) else tool.func
                                )
                            ]
                        ))
                    elif isinstance(tool, AgentTool):
                        t = tool._get_declaration() or genai_types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description
                        )
                        funcs.append(genai_types.Tool(
                            function_declarations=[t]
                        ))
                    elif isinstance(tool, BaseToolset):
                        toolset_tools = await tool.get_tools_with_prefix()
                        funcs.extend(await get_tool_functions(toolset_tools))
                return funcs

            agent = agent_loader.load_agent(agent_name) # type: ignore
            if isinstance(agent, App):
                agent = agent.root_agent

            tools = getattr(agent, "tools", [])
            tools_dict_list = [t.model_dump(exclude_none=True) for t in  await get_tool_functions(tools)]
            agent_eval_info = {
                "name": agent.name,
                "description": str(getattr(agent, "description", None)),
                "instruction": str(getattr(agent, "instruction", None)),
                "tool_declarations": tools_dict_list
            }
            return agent_eval_info

    if a2a:
        from starlette.middleware.base import BaseHTTPMiddleware
        from a2a_utils import a2a_card_dispatch # type: ignore
        app.add_middleware(
            BaseHTTPMiddleware, # type: ignore
            dispatch=a2a_card_dispatch
        )
    for fd in files_to_delete:
        fd.unlink()
    for fd in folders_to_delete:
        try:
            fd.rmdir()
        except OSError:
            pass

    if trace_to_cloud or otel_to_cloud:
        try:
            from opentelemetry import propagate
            from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
            from opentelemetry.propagators.textmap import Getter, default_getter

            class OriginalTracePropagator(TraceContextTextMapPropagator):
                """
                Custom propagator to extract the original traceparent from the x-original-traceparent header.
                This is needed because Cloud Run may rewrite the traceparent header.
                """
                def extract(self, carrier, context=None, getter: Getter = default_getter):
                    orig = getter.get(carrier, "x-original-traceparent")
                    if orig:
                        # We found the original traceparent.
                        # We pass a modified carrier (or a fake one) to the parent extractor.
                        # Usually, orig is a list, so we take the first element if it's there.
                        val = orig[0] if isinstance(orig, list) else orig
                        # We trick the standard TraceContext extractor by feeding it our value
                        return super().extract({"traceparent": val}, context)
                    # If not found, proceed with standard traceparent extraction
                    return super().extract(carrier, context, getter=getter)

            # Wrap the app with OpenTelemetryMiddleware to ensure
            # that the traceparent header is extracted and used for the parent span
            app = OpenTelemetryMiddleware(app)

            # Set this as our global propagator
            propagate.set_global_textmap(OriginalTracePropagator())
        except ImportError:
            print(
                "ERROR: Missing `opentelemetry-instrumentation-asgi` package."
            )


    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )
    server = uvicorn.Server(config)
    server.run()


################################################################################
if __name__ == "__main__":
    main()
