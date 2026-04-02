import asyncio, miniconf, aiomqtt, logging

async def main():
    async with miniconf.Client("10.42.0.1", protocol=aiomqtt.ProtocolVersion.V5, logger=logging.getLogger()) as client:
        discovered = await miniconf.discover(client, "dt/sinara/dual-iir/+")
        prefix = list(discovered.keys())[0] if discovered else "dt/sinara/dual-iir/04-91-62-d9-4c-7f"
        print("Discovered prefix:", prefix)
        interface = miniconf.Miniconf(client, prefix)
        try:
            print("Trying with leading slash:")
            await interface.set("/dual_iir/trigger", True)
            print("Success!")
        except Exception as e:
            print("Failed slash:", repr(e))
        try:
            print("Trying without leading slash:")
            await interface.set("dual_iir/trigger", True)
            print("Success!")
        except Exception as e:
            print("Failed no slash:", repr(e))

asyncio.run(main())
